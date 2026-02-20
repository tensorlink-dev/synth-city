"""
Leaderboard — tracks model scores over time with rolling averages and
emission weight allocation, matching the SN50 validator's moving average logic.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from live_validator.scoring import (
    ASSET_COEFFICIENTS,
    ScoringConfig,
    compute_softmax,
)

logger = logging.getLogger(__name__)


@dataclass
class ScoreEntry:
    """A single scoring result for one model on one asset at one time."""
    model_name: str
    asset: str
    scored_time: float       # Unix timestamp
    raw_crps: float
    prompt_score: float | None
    emission_weight: float
    scoring_mode: str        # "low" or "high"


@dataclass
class ModelStats:
    """Aggregated statistics for a model."""
    model_name: str
    total_scores: int = 0
    rolling_avg_crps: float = float("inf")
    rolling_avg_prompt_score: float = float("inf")
    emission_weight: float = 0.0
    last_scored: float = 0.0
    best_crps: float = float("inf")
    worst_crps: float = 0.0
    per_asset_avg: dict[str, float] = field(default_factory=dict)


class Leaderboard:
    """Maintains a live leaderboard of model performance.

    Mirrors the SN50 validator's moving average and reward weight logic:
    - Rolling window of scores (configurable, default 10 days for low-freq)
    - Per-asset coefficient weighting
    - Softmax emission allocation
    """

    def __init__(
        self,
        models: list[str],
        save_path: Path | None = None,
        max_entries: int = 10000,
    ) -> None:
        self.models = models
        self.save_path = save_path
        self.max_entries = max_entries
        self.entries: list[ScoreEntry] = []
        self.stats: dict[str, ModelStats] = {
            m: ModelStats(model_name=m) for m in models
        }

    def add_scores(
        self,
        results: dict[str, dict],
        asset: str,
        scoring_config: ScoringConfig,
    ) -> None:
        """Add scoring results for all models for a single asset/round.

        Parameters
        ----------
        results : dict
            model_name -> {raw_crps, prompt_score, emission_weight, ...}
            as returned by scoring.score_models().
        asset : str
            The asset that was scored.
        scoring_config : ScoringConfig
            The scoring mode used.
        """
        ts = time.time()
        for model_name, data in results.items():
            entry = ScoreEntry(
                model_name=model_name,
                asset=asset,
                scored_time=ts,
                raw_crps=data["raw_crps"],
                prompt_score=data.get("prompt_score"),
                emission_weight=data.get("emission_weight", 0.0),
                scoring_mode=scoring_config.label,
            )
            self.entries.append(entry)

        # Trim old entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        # Recompute stats
        self._recompute_stats(scoring_config)

    def _recompute_stats(self, scoring_config: ScoringConfig) -> None:
        """Recompute rolling averages and emission weights for all models."""
        window_seconds = scoring_config.window_days * 86400
        cutoff = time.time() - window_seconds

        # Gather per-model, per-asset scores within the window
        model_asset_scores: dict[str, dict[str, list[float]]] = {
            m: {} for m in self.models
        }

        for entry in self.entries:
            if entry.scored_time < cutoff:
                continue
            if entry.scoring_mode != scoring_config.label:
                continue
            if entry.raw_crps < 0:
                continue
            model_scores = model_asset_scores[entry.model_name]
            if entry.asset not in model_scores:
                model_scores[entry.asset] = []
            model_scores[entry.asset].append(entry.raw_crps)

        # Compute weighted rolling average per model
        rolling_avgs: dict[str, float] = {}
        for model_name in self.models:
            asset_scores = model_asset_scores[model_name]
            if not asset_scores:
                rolling_avgs[model_name] = float("inf")
                continue

            weighted_sum = 0.0
            total_weight = 0.0
            per_asset_avg: dict[str, float] = {}

            for asset, scores in asset_scores.items():
                coeff = ASSET_COEFFICIENTS.get(asset, 1.0)
                avg = float(np.mean(scores))
                per_asset_avg[asset] = avg
                weighted_sum += avg * coeff
                total_weight += coeff

            rolling_avg = weighted_sum / total_weight if total_weight > 0 else float("inf")
            rolling_avgs[model_name] = rolling_avg

            # Update stats
            stats = self.stats[model_name]
            stats.rolling_avg_crps = rolling_avg
            stats.per_asset_avg = per_asset_avg
            stats.total_scores = sum(len(s) for s in asset_scores.values())
            stats.last_scored = max(
                (e.scored_time for e in self.entries if e.model_name == model_name),
                default=0.0,
            )

            all_crps = [c for scores in asset_scores.values() for c in scores]
            if all_crps:
                stats.best_crps = float(np.min(all_crps))
                stats.worst_crps = float(np.max(all_crps))

        # Compute softmax emission weights
        model_names = list(self.models)
        avg_array = np.array([rolling_avgs[m] for m in model_names])

        # Replace inf with a large penalty for softmax
        finite_mask = np.isfinite(avg_array)
        max_finite = float(np.max(avg_array[finite_mask])) if np.any(finite_mask) else 1.0
        avg_array = np.where(np.isfinite(avg_array), avg_array, max_finite * 2)

        weights = compute_softmax(avg_array, scoring_config.softmax_beta)
        for i, name in enumerate(model_names):
            self.stats[name].emission_weight = float(weights[i])

    def get_ranking(self) -> list[dict]:
        """Return models ranked by rolling average CRPS (best first)."""
        ranked = sorted(self.stats.values(), key=lambda s: s.rolling_avg_crps)
        return [
            {
                "rank": i + 1,
                "model": s.model_name,
                "rolling_avg_crps": s.rolling_avg_crps,
                "emission_weight": s.emission_weight,
                "total_scores": s.total_scores,
                "best_crps": s.best_crps,
                "per_asset": s.per_asset_avg,
                "last_scored": datetime.fromtimestamp(
                    s.last_scored, tz=timezone.utc
                ).isoformat() if s.last_scored > 0 else "never",
            }
            for i, s in enumerate(ranked)
        ]

    def format_table(self) -> str:
        """Format the leaderboard as a readable table string."""
        ranking = self.get_ranking()
        if not ranking:
            return "No scores recorded yet."

        lines = [
            "",
            "=" * 90,
            f"{'Rank':<6} {'Model':<20} {'Avg CRPS':<14} {'Emission %':<12} "
            f"{'Best CRPS':<14} {'Scores':<8} {'Last Scored'}",
            "-" * 90,
        ]
        for r in ranking:
            avg_crps = f"{r['rolling_avg_crps']:.4f}" if r["rolling_avg_crps"] < 1e6 else "N/A"
            best_crps = f"{r['best_crps']:.4f}" if r["best_crps"] < 1e6 else "N/A"
            emission = f"{r['emission_weight'] * 100:.1f}%"
            lines.append(
                f"  {r['rank']:<4} {r['model']:<20} {avg_crps:<14} {emission:<12} "
                f"{best_crps:<14} {r['total_scores']:<8} {r['last_scored']}"
            )
        lines.append("=" * 90)
        return "\n".join(lines)

    def save(self) -> None:
        """Persist leaderboard state to disk."""
        if self.save_path is None:
            return
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "ranking": self.get_ranking(),
            "entries_count": len(self.entries),
            "timestamp": time.time(),
        }
        self.save_path.write_text(json.dumps(data, indent=2, default=str))
        logger.debug("Saved leaderboard to %s", self.save_path)

    def load(self) -> None:
        """Load leaderboard state from disk (ranking only, not full history)."""
        if self.save_path is None or not self.save_path.exists():
            return
        logger.info("Loading leaderboard from %s", self.save_path)
