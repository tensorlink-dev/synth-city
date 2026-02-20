"""
Live validator runner — continuously runs the top 3 models, fetches live prices,
and scores predictions exactly as the SN50 validator would.

This is a "streamlined" local validator: since all models run on the same machine,
we skip the Bittensor network layer and directly generate + score predictions.

Flow:
    1. Fetch live prices from Pyth → fit models on recent history
    2. Generate 1000 Monte Carlo paths per model per asset
    3. Record price snapshots every `time_increment` seconds
    4. After the prediction window elapses, score predictions against reality
    5. Update the leaderboard with rolling averages + emission weights
    6. Repeat
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import SN50_ASSETS, SN50_NUM_PATHS, WORKSPACE_DIR
from data.market import get_close_prices
from live_validator.leaderboard import Leaderboard
from live_validator.price_feed import PriceFeed
from live_validator.scoring import (
    HIGH_FREQUENCY,
    LOW_FREQUENCY,
    ScoringConfig,
    score_models,
)
from models.base import BaseForecaster
from models.garch import GARCHForecaster
from models.gbm import GBMForecaster
from models.stochastic_vol import HestonForecaster

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry — the top 3 models we run as virtual miners
# ---------------------------------------------------------------------------

def build_models() -> dict[str, BaseForecaster]:
    """Instantiate the 3 competing models."""
    return {
        "GBM": GBMForecaster(),
        "GARCH(1,1)-t": GARCHForecaster(variant="GARCH", p=1, q=1, dist="t"),
        "Heston-SV": HestonForecaster(),
    }


@dataclass
class PendingPrediction:
    """A prediction waiting to be scored once the window elapses."""
    model_name: str
    asset: str
    paths: np.ndarray           # (num_sims, num_steps)
    start_time: float           # Unix timestamp
    scoring_config: ScoringConfig
    num_steps: int

    @property
    def end_time(self) -> float:
        return self.start_time + self.scoring_config.time_length

    @property
    def is_ready_to_score(self) -> bool:
        return time.time() >= self.end_time


@dataclass
class ValidatorState:
    """Runtime state for the live validator."""
    models: dict[str, BaseForecaster] = field(default_factory=dict)
    pending: list[PendingPrediction] = field(default_factory=list)
    prices_at_start: dict[str, float] = field(default_factory=dict)
    last_prediction_time: float = 0.0
    last_fit_time: float = 0.0
    rounds_completed: int = 0
    total_scores: int = 0
    running: bool = True


class LiveValidator:
    """Main live validator that runs models and scores them continuously.

    Parameters
    ----------
    assets : list of str
        Assets to predict. Defaults to all SN50 assets.
    mode : str
        "low" for 24h predictions or "high" for 1h. Default "low".
    prediction_interval : int
        Seconds between prediction rounds. Default 3600 (1 hour).
    price_poll_interval : int
        Seconds between price recordings. Default 60.
    refit_interval : int
        Seconds between model refits. Default 3600.
    save_dir : Path or None
        Directory for persisting leaderboard state.
    """

    def __init__(
        self,
        assets: list[str] | None = None,
        mode: str = "low",
        prediction_interval: int = 3600,
        price_poll_interval: int = 60,
        refit_interval: int = 3600,
        save_dir: Path | None = None,
    ) -> None:
        self.assets = assets or list(SN50_ASSETS.keys())
        self.scoring_config = LOW_FREQUENCY if mode == "low" else HIGH_FREQUENCY
        self.prediction_interval = prediction_interval
        self.price_poll_interval = price_poll_interval
        self.refit_interval = refit_interval
        self.save_dir = save_dir or (WORKSPACE_DIR / "live_validator")

        self.price_feed = PriceFeed(
            assets=self.assets,
            max_history=2000,  # Keep plenty of history for scoring
        )
        self.state = ValidatorState(models=build_models())
        self.leaderboard = Leaderboard(
            models=list(self.state.models.keys()),
            save_path=self.save_dir / "leaderboard.json",
        )

        # Graceful shutdown
        self._shutdown = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame: object) -> None:
        logger.info("Received signal %d — shutting down gracefully...", signum)
        self._shutdown = True
        self.state.running = False

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def fit_models(self) -> None:
        """Fit all models on recent historical price data."""
        logger.info("Fitting models on historical data...")
        for asset in self.assets:
            try:
                prices = get_close_prices(asset, days=30)
                if len(prices) < 20:
                    logger.warning(
                        "Not enough data for %s (%d points), skipping",
                        asset, len(prices),
                    )
                    continue
                for name, model in self.state.models.items():
                    try:
                        model.fit(prices)
                        logger.info("  Fitted %s on %s (%d data points)", name, asset, len(prices))
                    except Exception as exc:
                        logger.error("  Failed to fit %s on %s: %s", name, asset, exc)
            except Exception as exc:
                logger.warning("Could not fetch data for %s: %s", asset, exc)
        self.state.last_fit_time = time.time()

    def fit_models_from_feed(self) -> None:
        """Refit models using accumulated price feed data (for longer runs)."""
        for asset in self.assets:
            hist = self.price_feed.get_historical_prices(asset, count=200)
            if hist is None:
                continue
            for name, model in self.state.models.items():
                try:
                    model.fit(hist)
                except Exception as exc:
                    logger.error("Refit failed for %s on %s: %s", name, asset, exc)
        self.state.last_fit_time = time.time()

    # ------------------------------------------------------------------
    # Prediction generation
    # ------------------------------------------------------------------

    def generate_predictions(self) -> int:
        """Generate predictions from all models for all assets.

        Returns the number of predictions generated.
        """
        num_steps = self.scoring_config.time_length // self.scoring_config.time_increment + 1
        start_time = time.time()
        count = 0

        # Fetch current prices for s0
        current_prices = self.price_feed.fetch_all_prices()
        self.state.prices_at_start = current_prices

        for asset in self.assets:
            s0 = current_prices.get(asset)
            if s0 is None:
                logger.warning("No live price for %s — skipping prediction", asset)
                continue

            for model_name, model in self.state.models.items():
                try:
                    paths = model.generate_paths(
                        asset=asset,
                        num_paths=SN50_NUM_PATHS,
                        num_steps=num_steps,
                        s0=s0,
                    )
                    assert paths.shape == (SN50_NUM_PATHS, num_steps), (
                        f"Shape mismatch: {paths.shape}"
                    )

                    self.state.pending.append(PendingPrediction(
                        model_name=model_name,
                        asset=asset,
                        paths=paths,
                        start_time=start_time,
                        scoring_config=self.scoring_config,
                        num_steps=num_steps,
                    ))
                    count += 1
                except Exception as exc:
                    logger.error(
                        "Prediction failed for %s/%s: %s", model_name, asset, exc
                    )

        self.state.last_prediction_time = start_time
        logger.info(
            "Generated %d predictions (%d models x %d assets) at %s",
            count,
            len(self.state.models),
            len(self.assets),
            datetime.fromtimestamp(start_time, tz=timezone.utc).strftime("%H:%M:%S UTC"),
        )
        return count

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_ready_predictions(self) -> int:
        """Score any predictions whose windows have elapsed.

        Returns the number of predictions scored.
        """
        ready = [p for p in self.state.pending if p.is_ready_to_score]
        if not ready:
            return 0

        # Group by (asset, start_time) so we score all models together
        groups: dict[tuple[str, float], list[PendingPrediction]] = {}
        for pred in ready:
            key = (pred.asset, pred.start_time)
            if key not in groups:
                groups[key] = []
            groups[key].append(pred)

        scored_count = 0
        for (asset, start_time), preds in groups.items():
            config = preds[0].scoring_config
            num_steps = preds[0].num_steps

            # Get real price path from feed
            real_path = self.price_feed.get_price_path(
                asset=asset,
                start_time=start_time,
                time_increment=config.time_increment,
                num_steps=num_steps,
            )

            if real_path is None:
                logger.warning(
                    "Cannot score %s predictions from %.0f — insufficient price history",
                    asset, start_time,
                )
                # Remove these predictions, they can never be scored
                for pred in preds:
                    self.state.pending.remove(pred)
                continue

            # Build model predictions dict for scoring
            model_preds = {pred.model_name: pred.paths for pred in preds}

            # Score
            results = score_models(model_preds, real_path, config)

            # Log results
            logger.info("Scored %s (started %s):", asset,
                        datetime.fromtimestamp(start_time, tz=timezone.utc).strftime("%H:%M:%S"))
            for name, data in results.items():
                crps = data["raw_crps"]
                weight = data["emission_weight"]
                crps_str = f"{crps:.4f}" if crps >= 0 else "FAILED"
                logger.info("  %-20s  CRPS=%-12s  emission=%.1f%%", name, crps_str, weight * 100)

            # Update leaderboard
            self.leaderboard.add_scores(results, asset, config)

            # Remove scored predictions
            for pred in preds:
                self.state.pending.remove(pred)

            scored_count += len(preds)

        self.state.total_scores += scored_count
        self.state.rounds_completed += 1
        return scored_count

    # ------------------------------------------------------------------
    # Status display
    # ------------------------------------------------------------------

    def print_status(self) -> None:
        """Print current validator status and leaderboard."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        pending_count = len(self.state.pending)
        oldest_pending = ""
        if self.state.pending:
            remaining = max(0, self.state.pending[0].end_time - time.time())
            oldest_pending = f" (oldest: {remaining:.0f}s until scorable)"

        logger.info(
            "\n[%s] Round %d | %d total scores | %d pending%s",
            now, self.state.rounds_completed, self.state.total_scores,
            pending_count, oldest_pending,
        )
        logger.info(self.leaderboard.format_table())

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the live validator continuously."""
        logger.info("=" * 70)
        logger.info("LIVE VALIDATOR — scoring top 3 models on SN50 assets")
        logger.info("Mode: %s frequency", self.scoring_config.label)
        logger.info("Assets: %s", ", ".join(self.assets))
        logger.info("Models: %s", ", ".join(self.state.models.keys()))
        logger.info("Prediction interval: %ds", self.prediction_interval)
        logger.info("Price poll interval: %ds", self.price_poll_interval)
        logger.info("Prediction window: %ds (%s)",
                     self.scoring_config.time_length,
                     f"{self.scoring_config.time_length // 3600}h"
                     if self.scoring_config.time_length >= 3600
                     else f"{self.scoring_config.time_length // 60}m")
        logger.info("=" * 70)

        # Step 1: Initial model fitting from historical data
        self.fit_models()

        # Step 2: Seed price feed with initial recordings
        logger.info("Recording initial prices...")
        self.price_feed.record_prices()

        # Step 3: Generate first predictions
        self.generate_predictions()

        last_poll = time.time()
        last_status = time.time()
        status_interval = 120  # Print status every 2 minutes

        logger.info(
            "Now recording live prices every %ds. "
            "Predictions will be scored after %ds window elapses.",
            self.price_poll_interval,
            self.scoring_config.time_length,
        )

        while self.state.running and not self._shutdown:
            now = time.time()

            # Record prices
            if now - last_poll >= self.price_poll_interval:
                prices = self.price_feed.record_prices()
                last_poll = now
                if prices:
                    logger.debug("Recorded prices for %d assets", len(prices))

            # Score any ready predictions
            scored = self.score_ready_predictions()
            if scored > 0:
                self.print_status()
                self.leaderboard.save()

            # Generate new predictions on schedule
            if now - self.state.last_prediction_time >= self.prediction_interval:
                # Refit models if needed
                if now - self.state.last_fit_time >= self.refit_interval:
                    self.fit_models_from_feed()
                self.generate_predictions()

            # Periodic status
            if now - last_status >= status_interval:
                self.print_status()
                last_status = now

            # Sleep briefly
            time.sleep(min(5, self.price_poll_interval / 2))

        # Shutdown
        logger.info("Live validator stopped.")
        self.print_status()
        self.leaderboard.save()
        self.price_feed.close()


def run_live_validator(
    assets: list[str] | None = None,
    mode: str = "low",
    prediction_interval: int = 3600,
    price_poll_interval: int = 60,
    refit_interval: int = 3600,
) -> None:
    """Entry point for the live validator."""
    validator = LiveValidator(
        assets=assets,
        mode=mode,
        prediction_interval=prediction_interval,
        price_poll_interval=price_poll_interval,
        refit_interval=refit_interval,
    )
    validator.run()
