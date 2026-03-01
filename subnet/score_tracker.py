"""
Scoring emulator — local replica of the SN50 validator scoring loop.

Runs a background daemon that:
    1. Periodically generates prediction prompts (like a validator).
    2. Records model predictions with timestamps.
    3. Fetches realised prices as time elapses.
    4. Scores predictions at each CRPS evaluation increment.
    5. Persists everything to Hippius with a queryable key layout.

Hippius object layout::

    scores/prompts/{YYYY-MM-DD}/{prompt_id}.json   per-prompt record
    scores/daily/{YYYY-MM-DD}.json                  daily aggregate
    scores/leaderboard.json                         rolling 10-day summary

Usage (programmatic)::

    tracker = ScoreTracker()
    tracker.record_prompt(predictions, t0_prices)   # record a new prompt
    tracker.score_pending()                         # score any mature prompts
    tracker.daily_summary("2026-03-01")             # build daily summary

Usage (daemon)::

    daemon = ScoringDaemon(tracker, miner, interval_minutes=5)
    daemon.start()   # runs in background thread
    daemon.stop()
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from config import SN50_ASSETS, SN50_STEP_MINUTES
from subnet.config import CRPS_EVAL_INCREMENTS, LEADERBOARD_WINDOW_DAYS
from subnet.validator import evaluate_prediction

logger = logging.getLogger(__name__)

# Cap on in-memory completed records to prevent unbounded growth.
_MAX_COMPLETED: int = 500

# ---------------------------------------------------------------------------
# Hippius helpers (reuse the storage layer)
# ---------------------------------------------------------------------------


def _hippius_put(key: str, data: Any) -> bool:
    """Store JSON to Hippius. Returns True on success."""
    try:
        from pipeline.tools.hippius_store import _put_json
        return _put_json(key, data)
    except Exception as exc:
        logger.warning("Hippius put failed for %s: %s", key, exc)
        return False


def _hippius_get(key: str) -> Any | None:
    """Load JSON from Hippius."""
    try:
        from pipeline.tools.hippius_store import _get_json
        return _get_json(key)
    except Exception as exc:
        logger.warning("Hippius get failed for %s: %s", key, exc)
        return None


def _hippius_list(prefix: str, max_keys: int = 1000) -> list[str]:
    """List keys under a prefix."""
    try:
        from pipeline.tools.hippius_store import _list_keys
        return _list_keys(prefix, max_keys=max_keys)
    except Exception as exc:
        logger.warning("Hippius list failed for %s: %s", prefix, exc)
        return []


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------


def _fetch_live_price(asset: str) -> float | None:
    """Fetch the latest price for an asset from Pyth oracle."""
    try:
        from pipeline.tools.market_data import get_latest_price
        result = json.loads(get_latest_price(asset))
        if "error" in result:
            logger.warning("Price fetch failed for %s: %s", asset, result["error"])
            return None
        return float(result["price"])
    except Exception as exc:
        logger.warning("Price fetch exception for %s: %s", asset, exc)
        return None


def _fetch_all_prices() -> dict[str, float]:
    """Fetch current prices for all scoreable assets."""
    prices: dict[str, float] = {}
    for asset in SN50_ASSETS:
        price = _fetch_live_price(asset)
        if price is not None:
            prices[asset] = price
    return prices


# ---------------------------------------------------------------------------
# Prompt record — represents a single prediction prompt
# ---------------------------------------------------------------------------


class PromptRecord:
    """A single prediction prompt with its predictions and scores."""

    def __init__(
        self,
        prompt_id: str,
        timestamp: float,
        t0_prices: dict[str, float],
        predictions: dict[str, np.ndarray],
        model_name: str = "unknown",
    ) -> None:
        self.prompt_id = prompt_id
        self.timestamp = timestamp
        self.t0_prices = t0_prices
        self.predictions = predictions  # asset -> (num_paths, num_steps)
        self.model_name = model_name
        self.scores: dict[str, dict[str, float]] = {}  # asset -> {crps_5min: ..., ...}
        self.weighted_crps: float | None = None
        # Price snapshots collected over time: asset -> {step_idx: price}
        self.price_snapshots: dict[str, dict[int, float]] = defaultdict(dict)
        self.status: str = "pending"  # pending | partial | scored | error

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a dict (without raw paths — too large for storage)."""
        pred_meta = {}
        for asset, paths in self.predictions.items():
            pred_meta[asset] = {
                "num_paths": paths.shape[0],
                "num_steps": paths.shape[1],
                "mean_t0": float(np.mean(paths[:, 0])),
                "std_t0": float(np.std(paths[:, 0])),
                "mean_final": float(np.mean(paths[:, -1])),
                "std_final": float(np.std(paths[:, -1])),
            }
        return {
            "prompt_id": self.prompt_id,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(
                self.timestamp, tz=timezone.utc
            ).isoformat(),
            "t0_prices": self.t0_prices,
            "model_name": self.model_name,
            "prediction_meta": pred_meta,
            "scores": self.scores,
            "weighted_crps": self.weighted_crps,
            "status": self.status,
            "assets_predicted": list(self.predictions.keys()),
            "assets_scored": list(self.scores.keys()),
            "snapshots_collected": {
                asset: len(steps) for asset, steps in self.price_snapshots.items()
            },
        }


# ---------------------------------------------------------------------------
# ScoreTracker — core scoring engine
# ---------------------------------------------------------------------------


class ScoreTracker:
    """Tracks predictions and scores them against realised prices.

    Thread-safe: all mutations are guarded by ``_lock``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: list[PromptRecord] = []
        self._completed: list[PromptRecord] = []
        self._stats = {
            "prompts_recorded": 0,
            "prompts_scored": 0,
            "prompts_failed": 0,
            "total_weighted_crps": 0.0,
            "best_weighted_crps": float("inf"),
            "worst_weighted_crps": 0.0,
        }

    # -- Recording ----------------------------------------------------------

    def record_prompt(
        self,
        predictions: dict[str, np.ndarray],
        t0_prices: dict[str, float],
        model_name: str = "unknown",
        timestamp: float | None = None,
    ) -> str:
        """Record a new prediction prompt.

        Returns the prompt_id.
        """
        ts = timestamp if timestamp is not None else time.time()
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        prompt_id = f"{dt.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

        record = PromptRecord(
            prompt_id=prompt_id,
            timestamp=ts,
            t0_prices=t0_prices,
            predictions=predictions,
            model_name=model_name,
        )

        with self._lock:
            self._pending.append(record)
            self._stats["prompts_recorded"] += 1

        logger.info(
            "Recorded prompt %s: %d assets, model=%s",
            prompt_id, len(predictions), model_name,
        )
        return prompt_id

    # -- Scoring ------------------------------------------------------------

    def score_prompt_with_realized(
        self,
        record: PromptRecord,
        realized: dict[str, np.ndarray],
    ) -> None:
        """Score a prompt against realised price data.

        Parameters
        ----------
        realized : dict
            asset -> np.ndarray of realised prices (num_steps,).
        """
        weighted_sum = 0.0
        total_weight = 0.0

        for asset, weight in SN50_ASSETS.items():
            if asset not in record.predictions or asset not in realized:
                continue

            paths = record.predictions[asset]
            real = realized[asset]

            try:
                scores = evaluate_prediction(paths, real, step_minutes=SN50_STEP_MINUTES)
                record.scores[asset] = scores
                weighted_sum += scores["crps_sum"] * weight
                total_weight += weight
            except Exception as exc:
                logger.warning(
                    "Scoring failed for %s in prompt %s: %s",
                    asset, record.prompt_id, exc,
                )
                record.scores[asset] = {"error": str(exc)}

        if total_weight > 0:
            record.weighted_crps = weighted_sum
            record.status = "scored"
        else:
            record.status = "error"

    def collect_price_snapshot(
        self,
        record: PromptRecord,
        current_prices: dict[str, float],
    ) -> None:
        """Capture a price snapshot for the current elapsed step.

        Stores the price at the appropriate 5-minute step index so that
        when the full horizon elapses, we have the correct realised price
        at each evaluation increment.
        """
        elapsed_minutes = (time.time() - record.timestamp) / 60.0
        current_step = int(elapsed_minutes) // SN50_STEP_MINUTES

        for asset in record.predictions:
            if asset not in current_prices:
                continue
            # Store price for this step if we haven't already
            if current_step not in record.price_snapshots[asset]:
                record.price_snapshots[asset][current_step] = current_prices[asset]

    def score_prompt_deferred(self, record: PromptRecord) -> bool:
        """Score a prompt once the full 24h horizon has elapsed.

        Uses the collected price snapshots to build the realised price
        array and scores with ``evaluate_prediction`` — the same function
        the SN50 validator uses.

        Returns True if the prompt was scored in this call.
        """
        elapsed_minutes = (time.time() - record.timestamp) / 60.0
        max_increment = CRPS_EVAL_INCREMENTS[-1]  # 1440 min = 24h

        if elapsed_minutes < max_increment:
            # Mark as partial if we have any snapshots
            for asset in record.predictions:
                if record.price_snapshots.get(asset):
                    record.status = "partial"
                    break
            return False

        # Build realised price arrays from snapshots
        weighted_sum = 0.0
        total_weight = 0.0

        for asset, weight in SN50_ASSETS.items():
            if asset not in record.predictions:
                continue

            paths = record.predictions[asset]
            num_steps = paths.shape[1]
            snapshots = record.price_snapshots.get(asset, {})

            if not snapshots:
                continue

            # Build the realised price array from snapshots, forward-filling gaps
            realized = np.empty(num_steps)
            last_price = record.t0_prices.get(asset, 0.0)
            realized[0] = last_price

            for step in range(1, num_steps):
                if step in snapshots:
                    last_price = snapshots[step]
                realized[step] = last_price

            try:
                scores = evaluate_prediction(paths, realized, step_minutes=SN50_STEP_MINUTES)
                record.scores[asset] = scores
                weighted_sum += scores["crps_sum"] * weight
                total_weight += weight
            except Exception as exc:
                logger.warning(
                    "Deferred scoring failed for %s in prompt %s: %s",
                    asset, record.prompt_id, exc,
                )
                record.scores[asset] = {"error": str(exc)}

        if total_weight > 0:
            record.weighted_crps = weighted_sum
            record.status = "scored"
        else:
            record.status = "error"

        return record.status == "scored"

    def score_pending(self) -> int:
        """Score all pending/partial prompts using current prices.

        Collects price snapshots for all pending prompts, then attempts
        deferred scoring for any prompts that have reached the full 24h
        horizon.

        Returns the number of prompts that were fully scored.
        """
        current_prices = _fetch_all_prices()
        if not current_prices:
            logger.warning("No prices available — skipping scoring cycle")
            return 0

        scored_count = 0
        newly_completed: list[PromptRecord] = []

        with self._lock:
            # Collect price snapshots for all pending prompts
            for record in self._pending:
                self.collect_price_snapshot(record, current_prices)

            # Attempt deferred scoring for mature prompts
            still_pending: list[PromptRecord] = []
            for record in self._pending:
                scored = self.score_prompt_deferred(record)
                if scored:
                    scored_count += 1

                if record.status == "scored":
                    newly_completed.append(record)
                    self._completed.append(record)
                    self._stats["prompts_scored"] += 1
                    if record.weighted_crps is not None:
                        self._stats["total_weighted_crps"] += record.weighted_crps
                        self._stats["best_weighted_crps"] = min(
                            self._stats["best_weighted_crps"], record.weighted_crps
                        )
                        self._stats["worst_weighted_crps"] = max(
                            self._stats["worst_weighted_crps"], record.weighted_crps
                        )
                elif record.status == "error":
                    self._stats["prompts_failed"] += 1
                else:
                    still_pending.append(record)

            self._pending = still_pending

            # Trim completed to prevent unbounded memory growth
            if len(self._completed) > _MAX_COMPLETED:
                self._completed = self._completed[-_MAX_COMPLETED:]

        # Persist completed prompts to Hippius (outside lock — I/O is slow)
        for record in newly_completed:
            self._save_prompt(record)

        return scored_count

    # -- Hippius persistence ------------------------------------------------

    def _save_prompt(self, record: PromptRecord) -> None:
        """Persist a prompt record to Hippius."""
        dt = datetime.fromtimestamp(record.timestamp, tz=timezone.utc)
        date_str = dt.strftime("%Y-%m-%d")
        key = f"scores/prompts/{date_str}/{record.prompt_id}.json"
        _hippius_put(key, record.to_dict())

    def save_daily_summary(self, date_str: str | None = None) -> dict[str, Any]:
        """Build and persist a daily scoring summary.

        Parameters
        ----------
        date_str : str, optional
            Date in YYYY-MM-DD format. Defaults to today (UTC).
        """
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Collect all prompts for this date from completed + pending
        day_records: list[PromptRecord] = []
        with self._lock:
            for rec in self._completed + self._pending:
                dt = datetime.fromtimestamp(rec.timestamp, tz=timezone.utc)
                if dt.strftime("%Y-%m-%d") == date_str:
                    day_records.append(rec)

        if not day_records:
            # Try loading from Hippius
            return self._load_daily_from_hippius(date_str)

        # Build per-asset aggregates
        per_asset: dict[str, dict[str, Any]] = {}
        scored_crps_values: list[float] = []

        for rec in day_records:
            for asset, scores in rec.scores.items():
                if not isinstance(scores, dict):
                    continue
                if asset not in per_asset:
                    per_asset[asset] = {
                        "prompt_count": 0,
                        "crps_values": [],
                        "best_crps_sum": float("inf"),
                        "worst_crps_sum": 0.0,
                    }
                entry = per_asset[asset]
                entry["prompt_count"] += 1
                crps_sum = scores.get("crps_sum")
                if crps_sum is not None and isinstance(crps_sum, (int, float)):
                    entry["crps_values"].append(crps_sum)
                    entry["best_crps_sum"] = min(entry["best_crps_sum"], crps_sum)
                    entry["worst_crps_sum"] = max(entry["worst_crps_sum"], crps_sum)

        # Clean up for serialization
        for asset, entry in per_asset.items():
            vals = entry.pop("crps_values")
            if vals:
                entry["mean_crps_sum"] = float(np.mean(vals))
                entry["median_crps_sum"] = float(np.median(vals))
                scored_crps_values.extend(vals)
            else:
                entry["mean_crps_sum"] = None
                entry["median_crps_sum"] = None

        # Weighted CRPS values
        weighted_values = [
            r.weighted_crps for r in day_records
            if r.weighted_crps is not None
        ]

        summary = {
            "date": date_str,
            "total_prompts": len(day_records),
            "scored_prompts": sum(1 for r in day_records if r.status == "scored"),
            "pending_prompts": sum(1 for r in day_records if r.status in ("pending", "partial")),
            "failed_prompts": sum(1 for r in day_records if r.status == "error"),
            "per_asset": per_asset,
            "weighted_crps": {
                "mean": float(np.mean(weighted_values)) if weighted_values else None,
                "median": float(np.median(weighted_values)) if weighted_values else None,
                "best": float(min(weighted_values)) if weighted_values else None,
                "worst": float(max(weighted_values)) if weighted_values else None,
                "count": len(weighted_values),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        key = f"scores/daily/{date_str}.json"
        _hippius_put(key, summary)
        return summary

    def save_leaderboard(self) -> dict[str, Any]:
        """Build and persist the rolling leaderboard.

        Aggregates scores over the LEADERBOARD_WINDOW_DAYS window.
        """
        # Gather daily summaries for the window
        now = datetime.now(timezone.utc)
        daily_summaries: list[dict[str, Any]] = []

        for days_ago in range(LEADERBOARD_WINDOW_DAYS):
            date_str = (now - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            summary = _hippius_get(f"scores/daily/{date_str}.json")
            if summary:
                daily_summaries.append(summary)

        # Aggregate
        total_prompts = sum(s.get("total_prompts", 0) for s in daily_summaries)
        scored_prompts = sum(s.get("scored_prompts", 0) for s in daily_summaries)

        all_weighted = []
        for s in daily_summaries:
            wc = s.get("weighted_crps", {})
            if wc and wc.get("mean") is not None:
                all_weighted.append(wc["mean"])

        # Per-asset rolling stats
        per_asset_rolling: dict[str, dict[str, Any]] = {}
        for s in daily_summaries:
            for asset, stats in s.get("per_asset", {}).items():
                if asset not in per_asset_rolling:
                    per_asset_rolling[asset] = {"crps_values": [], "prompt_count": 0}
                per_asset_rolling[asset]["prompt_count"] += stats.get("prompt_count", 0)
                mean_val = stats.get("mean_crps_sum")
                if mean_val is not None:
                    per_asset_rolling[asset]["crps_values"].append(mean_val)

        for asset, entry in per_asset_rolling.items():
            vals = entry.pop("crps_values")
            if vals:
                entry["mean_crps"] = float(np.mean(vals))
                entry["best_daily_crps"] = float(min(vals))
                entry["trend"] = "improving" if len(vals) > 1 and vals[0] < vals[-1] else "flat"
            else:
                entry["mean_crps"] = None

        leaderboard = {
            "window_days": LEADERBOARD_WINDOW_DAYS,
            "days_with_data": len(daily_summaries),
            "total_prompts": total_prompts,
            "scored_prompts": scored_prompts,
            "weighted_crps": {
                "rolling_mean": float(np.mean(all_weighted)) if all_weighted else None,
                "rolling_best": float(min(all_weighted)) if all_weighted else None,
                "rolling_worst": float(max(all_weighted)) if all_weighted else None,
            },
            "per_asset": per_asset_rolling,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        _hippius_put("scores/leaderboard.json", leaderboard)
        return leaderboard

    def _load_daily_from_hippius(self, date_str: str) -> dict[str, Any]:
        """Load a daily summary from Hippius if it exists."""
        data = _hippius_get(f"scores/daily/{date_str}.json")
        if data:
            return data
        return {"date": date_str, "total_prompts": 0, "status": "no_data"}

    # -- Queries ------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return current tracker status."""
        with self._lock:
            return {
                "pending_prompts": len(self._pending),
                "completed_prompts": len(self._completed),
                **self._stats,
                "pending_details": [
                    {
                        "prompt_id": r.prompt_id,
                        "status": r.status,
                        "age_minutes": (time.time() - r.timestamp) / 60,
                        "assets": list(r.predictions.keys()),
                        "scored_assets": list(r.scores.keys()),
                    }
                    for r in self._pending
                ],
            }

    def get_recent_scores(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent scored prompts."""
        with self._lock:
            records = sorted(
                self._completed + [r for r in self._pending if r.status == "partial"],
                key=lambda r: r.timestamp,
                reverse=True,
            )[:limit]
        return [r.to_dict() for r in records]

    def load_prompt_history(
        self,
        date_str: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Load scored prompts from Hippius for a given date."""
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        keys = _hippius_list(f"scores/prompts/{date_str}/", max_keys=limit)
        results: list[dict[str, Any]] = []
        for key in keys:
            data = _hippius_get(key)
            if data:
                results.append(data)

        # Sort by timestamp, most recent first
        results.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
        return results

    def load_leaderboard(self) -> dict[str, Any]:
        """Load the current leaderboard from Hippius."""
        data = _hippius_get("scores/leaderboard.json")
        return data or {"status": "no_data", "window_days": LEADERBOARD_WINDOW_DAYS}


# ---------------------------------------------------------------------------
# ScoringDaemon — background thread that drives the scoring loop
# ---------------------------------------------------------------------------


class ScoringDaemon:
    """Background daemon that continuously generates prompts and scores them.

    Mimics the SN50 validator loop:
        1. Every ``interval_minutes``, generate a new prompt.
        2. Continuously score pending prompts as prices mature.
        3. Persist everything to Hippius.
        4. Rebuild the daily summary and leaderboard periodically.
    """

    def __init__(
        self,
        tracker: ScoreTracker,
        miner: Any | None = None,
        interval_minutes: int = 5,
    ) -> None:
        self.tracker = tracker
        self.miner = miner  # SynthMiner instance (optional)
        self.interval_minutes = interval_minutes
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_prompt_time: float = 0.0
        self._last_daily_time: float = 0.0

    def start(self) -> None:
        """Start the scoring daemon in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Scoring daemon is already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="scoring-daemon")
        self._thread.start()
        logger.info(
            "Scoring daemon started (interval=%dmin, miner=%s)",
            self.interval_minutes,
            "attached" if self.miner else "none",
        )

    def stop(self) -> None:
        """Stop the scoring daemon."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None
        logger.info("Scoring daemon stopped")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run_loop(self) -> None:
        """Main daemon loop."""
        logger.info("Scoring daemon loop started")

        while not self._stop_event.is_set():
            try:
                now = time.time()

                # Generate a new prompt if interval has elapsed
                if now - self._last_prompt_time >= self.interval_minutes * 60:
                    self._generate_prompt()
                    self._last_prompt_time = now

                # Score any pending prompts
                scored = self.tracker.score_pending()
                if scored:
                    logger.info("Scored %d prompt(s) in this cycle", scored)

                # Rebuild daily summary every 30 minutes
                if now - self._last_daily_time >= 1800:
                    self.tracker.save_daily_summary()
                    self.tracker.save_leaderboard()
                    self._last_daily_time = now

            except Exception as exc:
                logger.error("Scoring daemon error: %s", exc, exc_info=True)

            # Sleep in small increments so stop is responsive
            for _ in range(60):  # check every second for 1 minute
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def _generate_prompt(self) -> None:
        """Generate a prediction prompt using the attached miner."""
        if self.miner is None:
            logger.debug("No miner attached — skipping prompt generation")
            return

        try:
            # Fetch current prices
            prices = _fetch_all_prices()
            if not prices:
                logger.warning("No prices available — skipping prompt generation")
                return

            # Update miner prices and generate predictions
            predictions: dict[str, np.ndarray] = {}
            for asset, price in prices.items():
                self.miner.set_price(asset, price)
                if asset in self.miner.models:
                    try:
                        pred = self.miner.generate_prediction(asset, horizon="24h")
                        predictions[asset] = np.array(pred["paths"])
                    except Exception as exc:
                        logger.warning("Prediction failed for %s: %s", asset, exc)

            if predictions:
                model_names = [type(m).__name__ for m in self.miner.models.values()]
                model_name = model_names[0] if len(set(model_names)) == 1 else "mixed"
                prompt_id = self.tracker.record_prompt(
                    predictions=predictions,
                    t0_prices=prices,
                    model_name=model_name,
                )
                logger.info(
                    "Generated prompt %s with %d asset predictions",
                    prompt_id, len(predictions),
                )
            else:
                logger.warning("No predictions generated — no models registered?")

        except Exception as exc:
            logger.error("Prompt generation failed: %s", exc, exc_info=True)
