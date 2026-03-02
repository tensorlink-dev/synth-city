"""
Validator backtest — replay historical prices through the SN50 scoring pipeline.

Loads 5-minute OHLCV data from the HuggingFace training dataset, slides a
24-hour window through time at a configurable prompt interval, generates
predictions via a SynthMiner, and scores each prompt with the same CRPS
evaluation used by the SN50 validator.

Usage (programmatic)::

    from subnet.backtest import ValidatorBacktest
    from subnet.miner import SynthMiner

    miner = SynthMiner()
    miner.register_model("BTC", my_btc_model)

    bt = ValidatorBacktest(miner=miner, interval_minutes=60, max_prompts=50)
    results = bt.run()
    print(results["summary"])

Usage (CLI)::

    synth-city score backtest                      # GBM baseline, all assets
    synth-city score backtest --assets BTC,ETH     # specific assets
    synth-city score backtest --interval 120       # prompt every 2h
    synth-city score backtest --max-prompts 100    # more evaluation points
    synth-city score backtest --num-paths 200      # paths per prediction
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from config import (
    HF_TRAINING_DATA_REPO,
    SN50_ASSETS,
    SN50_STEP_MINUTES,
    SN50_TO_HF_ASSET,
)
from subnet.config import NUM_STEPS_24H
from subnet.miner import SynthMiner
from subnet.score_tracker import PromptRecord, ScoreTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GBM baseline forecaster
# ---------------------------------------------------------------------------


class GBMForecaster:
    """Geometric Brownian Motion baseline for backtesting.

    Generates Monte Carlo paths using drift and volatility calibrated from
    historical price data.  Useful as a benchmark to compare trained models
    against.
    """

    def __init__(
        self,
        price_history: np.ndarray | None = None,
        mu: float = 0.0,
        sigma: float = 0.02,
    ) -> None:
        if price_history is not None and len(price_history) > 1:
            log_returns = np.diff(np.log(price_history))
            self.mu = float(np.mean(log_returns))
            self.sigma = float(np.std(log_returns))
            if self.sigma < 1e-10:
                self.sigma = 0.02
        else:
            self.mu = mu
            self.sigma = sigma

    def generate_paths(
        self,
        asset: str,
        num_paths: int,
        num_steps: int,
        s0: float | None = None,
    ) -> np.ndarray:
        if s0 is None or s0 <= 0:
            raise ValueError(f"GBMForecaster requires a positive s0, got {s0}")

        paths = np.zeros((num_paths, num_steps))
        paths[:, 0] = s0

        for t in range(1, num_steps):
            z = np.random.standard_normal(num_paths)
            paths[:, t] = paths[:, t - 1] * np.exp(
                (self.mu - 0.5 * self.sigma**2) + self.sigma * z
            )

        return paths


# ---------------------------------------------------------------------------
# Historical price loading
# ---------------------------------------------------------------------------


def _load_5m_close_prices(assets: list[str]) -> dict[str, np.ndarray]:
    """Load raw 5-minute close prices from HuggingFace parquet data.

    Returns dict mapping SN50 asset name -> 1-D numpy array of close prices.
    """
    from huggingface_hub import hf_hub_download

    prices: dict[str, np.ndarray] = {}

    for sn50_name in assets:
        hf_name = SN50_TO_HF_ASSET.get(sn50_name)
        if hf_name is None:
            logger.warning("No HF mapping for asset %s — skipping", sn50_name)
            continue

        parquet_path = f"data/{hf_name}/5m.parquet"
        try:
            local_path = hf_hub_download(
                repo_id=HF_TRAINING_DATA_REPO,
                filename=parquet_path,
                repo_type="dataset",
            )
            df = pd.read_parquet(local_path)

            # Handle varying column name conventions
            close_col: str | None = None
            for candidate in ("close", "Close", "CLOSE"):
                if candidate in df.columns:
                    close_col = candidate
                    break
            if close_col is None:
                logger.warning(
                    "No close column for %s, columns: %s", sn50_name, list(df.columns)
                )
                continue

            close = df[close_col].to_numpy(dtype=np.float64)
            close = close[~np.isnan(close)]

            if len(close) < NUM_STEPS_24H:
                logger.warning(
                    "Insufficient data for %s: %d points (need >= %d)",
                    sn50_name,
                    len(close),
                    NUM_STEPS_24H,
                )
                continue

            prices[sn50_name] = close
            logger.info("Loaded %s: %d price points (%.1f days)", sn50_name, len(close),
                        len(close) * SN50_STEP_MINUTES / (60 * 24))
        except Exception as exc:
            logger.warning("Failed to load %s: %s", sn50_name, exc)

    return prices


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------


class ValidatorBacktest:
    """Replay historical prices through the SN50 scoring pipeline.

    Slides a 24-hour window through historical 5-minute price data, generates
    predictions at each prompt interval, and scores them against the known
    realised prices using the same CRPS evaluation the SN50 validator uses.
    """

    def __init__(
        self,
        miner: SynthMiner,
        assets: list[str] | None = None,
        interval_minutes: int = 60,
        max_prompts: int = 50,
        num_paths: int = 100,
    ) -> None:
        self.miner = miner
        self.assets = assets or list(SN50_ASSETS.keys())
        self.interval_minutes = max(interval_minutes, SN50_STEP_MINUTES)
        self.max_prompts = max_prompts
        self.num_paths = num_paths
        self._tracker = ScoreTracker()

    def run(self) -> dict[str, Any]:
        """Execute the backtest.

        Returns a dict with ``summary`` (aggregate statistics) and
        ``prompts`` (per-prompt scores).
        """
        logger.info(
            "Starting backtest: assets=%s, interval=%dmin, max_prompts=%d, num_paths=%d",
            self.assets,
            self.interval_minutes,
            self.max_prompts,
            self.num_paths,
        )

        # 1. Load historical price data
        all_prices = _load_5m_close_prices(self.assets)
        if not all_prices:
            return {"error": "No historical price data available"}

        # Restrict to assets we actually loaded
        active_assets = [a for a in self.assets if a in all_prices]
        if not active_assets:
            return {"error": "No usable assets after loading data"}

        # 2. Determine common data range
        min_len = min(len(all_prices[a]) for a in active_assets)
        horizon_steps = NUM_STEPS_24H  # 289
        interval_steps = self.interval_minutes // SN50_STEP_MINUTES

        last_valid_start = min_len - horizon_steps
        if last_valid_start <= 0:
            return {
                "error": (
                    f"Not enough data for 24h horizon "
                    f"(need {horizon_steps}, have {min_len})"
                )
            }

        # 3. Build prompt start indices (most recent data first, then reverse)
        prompt_starts: list[int] = []
        idx = last_valid_start
        while idx >= 0 and len(prompt_starts) < self.max_prompts:
            prompt_starts.append(idx)
            idx -= interval_steps
        prompt_starts.reverse()

        logger.info(
            "Running %d prompts over %d price points (%d assets)",
            len(prompt_starts),
            min_len,
            len(active_assets),
        )

        # 4. Run each prompt
        prompt_results: list[dict[str, Any]] = []
        scored_count = 0
        t_start = time.time()

        for i, start_idx in enumerate(prompt_starts):
            result_entry = self._run_single_prompt(
                i, start_idx, horizon_steps, active_assets, all_prices
            )
            prompt_results.append(result_entry)
            if result_entry["status"] == "scored":
                scored_count += 1

            if (i + 1) % 10 == 0:
                logger.info(
                    "Backtest progress: %d/%d prompts", i + 1, len(prompt_starts)
                )

        elapsed = time.time() - t_start

        # 5. Build summary
        summary = self._build_summary(
            prompt_results, active_assets, scored_count, elapsed
        )

        return {"summary": summary, "prompts": prompt_results}

    # -- internal helpers ---------------------------------------------------

    def _run_single_prompt(
        self,
        prompt_index: int,
        start_idx: int,
        horizon_steps: int,
        active_assets: list[str],
        all_prices: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Generate predictions and score a single prompt."""
        t0_prices: dict[str, float] = {}
        realized: dict[str, np.ndarray] = {}
        predictions: dict[str, np.ndarray] = {}

        for asset in active_assets:
            price_series = all_prices[asset]
            t0_prices[asset] = float(price_series[start_idx])
            realized[asset] = price_series[start_idx : start_idx + horizon_steps]

        # Set miner prices
        for asset, price in t0_prices.items():
            self.miner.set_price(asset, price)

        # Generate predictions (call model directly for path-count control)
        for asset in active_assets:
            if asset not in self.miner.models:
                continue
            model = self.miner.models[asset]
            try:
                paths = model.generate_paths(
                    asset=asset,
                    num_paths=self.num_paths,
                    num_steps=horizon_steps,
                    s0=t0_prices[asset],
                )
                predictions[asset] = paths
            except Exception as exc:
                logger.warning(
                    "Prediction failed for %s at idx %d: %s", asset, start_idx, exc
                )

        if not predictions:
            return {
                "prompt_index": prompt_index,
                "data_index": start_idx,
                "weighted_crps": None,
                "status": "error",
                "per_asset": {},
            }

        # Create a PromptRecord and score it
        record = PromptRecord(
            prompt_id=f"bt-{prompt_index:04d}-{start_idx}",
            timestamp=float(start_idx * SN50_STEP_MINUTES * 60),
            t0_prices=t0_prices,
            predictions=predictions,
            model_name="backtest",
        )
        self._tracker.score_prompt_with_realized(record, realized)

        per_asset_crps: dict[str, float | None] = {}
        for asset, scores in record.scores.items():
            if isinstance(scores, dict) and "crps_sum" in scores:
                per_asset_crps[asset] = scores["crps_sum"]
            else:
                per_asset_crps[asset] = None

        return {
            "prompt_index": prompt_index,
            "data_index": start_idx,
            "weighted_crps": record.weighted_crps,
            "status": record.status,
            "per_asset": per_asset_crps,
        }

    def _build_summary(
        self,
        prompt_results: list[dict[str, Any]],
        active_assets: list[str],
        scored_count: int,
        elapsed: float,
    ) -> dict[str, Any]:
        """Aggregate prompt-level results into a summary."""
        crps_values = [
            r["weighted_crps"]
            for r in prompt_results
            if r["weighted_crps"] is not None
        ]

        per_asset_summary: dict[str, dict[str, Any]] = {}
        for asset in active_assets:
            asset_scores = [
                r["per_asset"][asset]
                for r in prompt_results
                if asset in r["per_asset"] and r["per_asset"][asset] is not None
            ]
            if asset_scores:
                per_asset_summary[asset] = {
                    "mean_crps": float(np.mean(asset_scores)),
                    "median_crps": float(np.median(asset_scores)),
                    "best_crps": float(min(asset_scores)),
                    "worst_crps": float(max(asset_scores)),
                    "std_crps": float(np.std(asset_scores)),
                    "num_scored": len(asset_scores),
                    "weight": SN50_ASSETS.get(asset, 0.0),
                }

        return {
            "total_prompts": len(prompt_results),
            "scored_prompts": scored_count,
            "failed_prompts": len(prompt_results) - scored_count,
            "elapsed_seconds": round(elapsed, 2),
            "interval_minutes": self.interval_minutes,
            "num_paths": self.num_paths,
            "assets": active_assets,
            "weighted_crps": {
                "mean": float(np.mean(crps_values)) if crps_values else None,
                "median": float(np.median(crps_values)) if crps_values else None,
                "best": float(min(crps_values)) if crps_values else None,
                "worst": float(max(crps_values)) if crps_values else None,
                "std": float(np.std(crps_values)) if crps_values else None,
            },
            "per_asset": per_asset_summary,
        }


# ---------------------------------------------------------------------------
# CLI helper — build a GBM-baseline miner for quick backtesting
# ---------------------------------------------------------------------------


def build_baseline_miner(
    assets: list[str] | None = None,
    calibration_points: int = 2000,
) -> SynthMiner:
    """Create a SynthMiner with GBM forecasters calibrated from historical data.

    Loads the most recent ``calibration_points`` prices for each asset to
    estimate drift and volatility, then registers a ``GBMForecaster`` per asset.
    """
    target_assets = assets or list(SN50_ASSETS.keys())
    prices = _load_5m_close_prices(target_assets)

    miner = SynthMiner()
    for asset, price_series in prices.items():
        # Use the last `calibration_points` prices for calibration
        if len(price_series) > calibration_points:
            cal_data = price_series[-calibration_points:]
        else:
            cal_data = price_series
        forecaster = GBMForecaster(price_history=cal_data)
        miner.register_model(asset, forecaster)  # type: ignore[arg-type]
        logger.info(
            "Registered GBM baseline for %s (mu=%.6f, sigma=%.6f)",
            asset,
            forecaster.mu,
            forecaster.sigma,
        )

    return miner
