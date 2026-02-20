"""
Leaderboard — tracks model scores over time with rolling averages.

Pure raw CRPS ranking: lower is better, no softmax/emission normalisation.
Supports dynamic model registration (baselines + research challengers).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from live_validator.scoring import ASSET_COEFFICIENTS, ScoringConfig

logger = logging.getLogger(__name__)


@dataclass
class ScoreEntry:
    """A single scoring result for one model on one asset at one time."""
    model_name: str
    asset: str
    scored_time: float       # Unix timestamp
    raw_crps: float
    scoring_mode: str        # "low" or "high"
    is_baseline: bool = False


@dataclass
class ModelStats:
    """Aggregated statistics for a model."""
    model_name: str
    is_baseline: bool = False
    total_scores: int = 0
    rolling_avg_crps: float = float("inf")
    last_scored: float = 0.0
    best_crps: float = float("inf")
    worst_crps: float = 0.0
    per_asset_avg: dict[str, float] = field(default_factory=dict)


class Leaderboard:
    """Live leaderboard of model performance ranked by raw CRPS.

    Supports dynamic model registration so research API challengers
    can join the board at any time.
    """

    def __init__(
        self,
        save_path: Path | None = None,
        max_entries: int = 10000,
    ) -> None:
        self.save_path = save_path
        self.max_entries = max_entries
        self.entries: list[ScoreEntry] = []
        self.stats: dict[str, ModelStats] = {}
        self._baselines: set[str] = set()

    def register_model(
        self, name: str, is_baseline: bool = False
    ) -> None:
        """Register a model on the leaderboard."""
        if name not in self.stats:
            self.stats[name] = ModelStats(
                model_name=name, is_baseline=is_baseline
            )
        if is_baseline:
            self._baselines.add(name)

    @property
    def model_names(self) -> list[str]:
        return list(self.stats.keys())

    def add_scores(
        self,
        results: dict[str, dict],
        asset: str,
        scoring_config: ScoringConfig,
    ) -> None:
        """Add scoring results for models on a single asset/round.

        Parameters
        ----------
        results : dict
            model_name -> {raw_crps, detailed}
        asset : str
            The asset that was scored.
        scoring_config : ScoringConfig
            The scoring mode used.
        """
        ts = time.time()
        for model_name, data in results.items():
            # Auto-register unknown models
            if model_name not in self.stats:
                self.register_model(model_name)

            entry = ScoreEntry(
                model_name=model_name,
                asset=asset,
                scored_time=ts,
                raw_crps=data["raw_crps"],
                scoring_mode=scoring_config.label,
                is_baseline=model_name in self._baselines,
            )
            self.entries.append(entry)

        # Trim old entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        self._recompute_stats(scoring_config)

    def _recompute_stats(self, scoring_config: ScoringConfig) -> None:
        """Recompute rolling averages for all models."""
        window_seconds = scoring_config.window_days * 86400
        cutoff = time.time() - window_seconds

        # Gather per-model, per-asset scores within the window
        model_asset_scores: dict[str, dict[str, list[float]]] = {
            m: {} for m in self.stats
        }

        for entry in self.entries:
            if entry.scored_time < cutoff:
                continue
            if entry.scoring_mode != scoring_config.label:
                continue
            if entry.raw_crps < 0:
                continue
            if entry.model_name not in model_asset_scores:
                model_asset_scores[entry.model_name] = {}
            bucket = model_asset_scores[entry.model_name]
            if entry.asset not in bucket:
                bucket[entry.asset] = []
            bucket[entry.asset].append(entry.raw_crps)

        for model_name, asset_scores in model_asset_scores.items():
            stats = self.stats[model_name]
            if not asset_scores:
                stats.rolling_avg_crps = float("inf")
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

            stats.rolling_avg_crps = (
                weighted_sum / total_weight if total_weight > 0
                else float("inf")
            )
            stats.per_asset_avg = per_asset_avg
            stats.total_scores = sum(
                len(s) for s in asset_scores.values()
            )
            stats.last_scored = max(
                (e.scored_time for e in self.entries
                 if e.model_name == model_name),
                default=0.0,
            )

            all_crps = [
                c for scores in asset_scores.values() for c in scores
            ]
            if all_crps:
                stats.best_crps = float(np.min(all_crps))
                stats.worst_crps = float(np.max(all_crps))

    def get_best_crps(self) -> float:
        """Return the best (lowest) rolling avg CRPS on the board."""
        if not self.stats:
            return float("inf")
        return min(s.rolling_avg_crps for s in self.stats.values())

    def get_leader(self) -> str | None:
        """Return the name of the current leader, or None."""
        if not self.stats:
            return None
        best = min(self.stats.values(), key=lambda s: s.rolling_avg_crps)
        return best.model_name if best.rolling_avg_crps < float("inf") else None

    def get_ranking(self) -> list[dict]:
        """Return models ranked by rolling average CRPS (best first)."""
        ranked = sorted(
            self.stats.values(), key=lambda s: s.rolling_avg_crps
        )
        return [
            {
                "rank": i + 1,
                "model": s.model_name,
                "is_baseline": s.is_baseline,
                "rolling_avg_crps": s.rolling_avg_crps,
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
            "=" * 85,
            f"{'Rank':<6} {'Model':<25} {'Type':<10} "
            f"{'Avg CRPS':<14} {'Best CRPS':<14} {'Scores':<8}",
            "-" * 85,
        ]
        for r in ranking:
            avg = (
                f"{r['rolling_avg_crps']:.4f}"
                if r["rolling_avg_crps"] < 1e6 else "N/A"
            )
            best = (
                f"{r['best_crps']:.4f}"
                if r["best_crps"] < 1e6 else "N/A"
            )
            tag = "baseline" if r["is_baseline"] else "research"
            lines.append(
                f"  {r['rank']:<4} {r['model']:<25} {tag:<10} "
                f"{avg:<14} {best:<14} {r['total_scores']:<8}"
            )
        lines.append("=" * 85)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise the full leaderboard state for the dashboard."""
        return {
            "ranking": self.get_ranking(),
            "entries_count": len(self.entries),
            "leader": self.get_leader(),
            "best_crps": self.get_best_crps(),
            "timestamp": time.time(),
            "last_updated": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    def save(self) -> None:
        """Persist leaderboard state to disk."""
        if self.save_path is None:
            return
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str)
        )
        logger.debug("Saved leaderboard to %s", self.save_path)

    def load(self) -> None:
        """Load leaderboard state from disk (ranking only)."""
        if self.save_path is None or not self.save_path.exists():
            return
        logger.info("Loading leaderboard from %s", self.save_path)
