"""
Adaptive meta-strategy — mutable pipeline recovery configuration with bounds
enforcement and run history tracking.

The ``MetaStrategy`` dataclass holds all parameters that control how the
orchestrator retries failed stages, escalates temperature, and detects stalls.
These values can be modified at runtime by agents (via meta-strategy tools)
and persist across pipeline runs.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STRATEGY_PATH = Path("pipeline/meta_strategy_config.json")
_HISTORY_PATH = Path("workspace/meta_strategy_history.jsonl")

# Safety bounds on all tuneable parameters
_BOUNDS: dict[str, tuple[float, float]] = {
    "max_retries": (1, 20),
    "base_temperature": (0.0, 1.0),
    "temperature_step": (0.0, 0.5),
    "stall_threshold": (1, 10),
    "cooldown_retries": (0, 10),
}


@dataclass
class MetaStrategy:
    """Mutable pipeline recovery configuration with safety bounds."""

    max_retries: int = 5
    base_temperature: float = 0.1
    temperature_step: float = 0.1
    stall_threshold: int = 1
    cooldown_retries: int = 2
    per_stage_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors: list[str] = []
        for param, (lo, hi) in _BOUNDS.items():
            val = getattr(self, param, None)
            if val is not None:
                if val < lo or val > hi:
                    errors.append(
                        f"{param}={val} out of bounds [{lo}, {hi}]"
                    )

        # Validate per-stage overrides
        for stage, overrides in self.per_stage_overrides.items():
            if not isinstance(overrides, dict):
                errors.append(
                    f"per_stage_overrides[{stage!r}] must be a dict, "
                    f"got {type(overrides).__name__}"
                )
                continue
            for param, val in overrides.items():
                if param in _BOUNDS:
                    lo, hi = _BOUNDS[param]
                    if not isinstance(val, (int, float)):
                        errors.append(
                            f"per_stage_overrides[{stage!r}].{param}={val!r} "
                            f"must be numeric"
                        )
                    elif val < lo or val > hi:
                        errors.append(
                            f"per_stage_overrides[{stage!r}].{param}={val} "
                            f"out of bounds [{lo}, {hi}]"
                        )
        return errors

    def for_stage(self, stage_name: str) -> dict[str, Any]:
        """Return effective parameters for a specific stage.

        Merges per-stage overrides on top of the global defaults.
        """
        base = {
            "max_retries": self.max_retries,
            "base_temperature": self.base_temperature,
            "temperature_step": self.temperature_step,
            "stall_threshold": self.stall_threshold,
            "cooldown_retries": self.cooldown_retries,
        }
        overrides = self.per_stage_overrides.get(stage_name, {})
        base.update(overrides)
        return base

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MetaStrategy:
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: Path | None = None) -> None:
        """Persist strategy to JSON."""
        config_path = path or _STRATEGY_PATH
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | None = None) -> MetaStrategy:
        """Load strategy from JSON, falling back to defaults."""
        config_path = path or _STRATEGY_PATH
        if not config_path.exists():
            return cls()
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            strategy = cls.from_dict(data)
            errors = strategy.validate()
            if errors:
                logger.warning(
                    "Meta-strategy has %d error(s), using defaults: %s",
                    len(errors),
                    "; ".join(errors),
                )
                return cls()
            return strategy
        except Exception as exc:
            logger.warning("Failed to load meta-strategy, using defaults: %s", exc)
            return cls()


def record_run_event(
    pipeline_run_id: str,
    stage: str,
    attempt: int,
    temperature: float,
    success: bool,
    crps: float | None = None,
    error: str | None = None,
    path: Path | None = None,
) -> None:
    """Append a single run event to the JSONL history log."""
    history_path = path or _HISTORY_PATH
    history_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": time.time(),
        "pipeline_run_id": pipeline_run_id,
        "stage": stage,
        "attempt": attempt,
        "temperature": temperature,
        "success": success,
    }
    if crps is not None:
        event["crps"] = crps
    if error is not None:
        event["error"] = error[:500]

    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def load_run_history(last_n: int = 100, path: Path | None = None) -> list[dict[str, Any]]:
    """Load the most recent *last_n* run events from the JSONL log."""
    history_path = path or _HISTORY_PATH
    if not history_path.exists():
        return []
    try:
        lines = history_path.read_text(encoding="utf-8").strip().splitlines()
        recent = lines[-last_n:] if last_n > 0 else lines
        return [json.loads(line) for line in recent if line.strip()]
    except Exception as exc:
        logger.warning("Failed to load run history: %s", exc)
        return []
