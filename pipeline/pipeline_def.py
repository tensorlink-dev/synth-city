"""
Declarative pipeline definition — data-driven stage sequencing with safety constraints.

Replaces the hardcoded agent chain in the orchestrator with a mutable, validated
list of ``StageSpec`` objects.  Protected stages (planner, trainer, check_debug)
cannot be removed or displaced, while new stages can be freely inserted at
designated positions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Position ordering — stages must respect this sequence
POSITION_ORDER: list[str] = ["plan", "execute", "validate", "publish", "post"]

_CONFIG_PATH = Path("pipeline/pipeline_config.json")


@dataclass
class StageSpec:
    """Specification for a single pipeline stage."""

    name: str
    agent_name: str
    protected: bool = False
    position: str = "post"  # plan | execute | validate | publish | post
    retry: bool = False
    optional: bool = False
    user_message: str = "Begin the task."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageSpec:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# The default pipeline matching the original hardcoded orchestrator sequence
DEFAULT_PIPELINE: list[StageSpec] = [
    StageSpec(
        name="planner",
        agent_name="planner",
        protected=True,
        position="plan",
        retry=False,
        user_message=(
            "Discover the available components and produce an experiment plan "
            "to find the best architecture for SN50 CRPS."
        ),
    ),
    StageSpec(
        name="trainer",
        agent_name="trainer",
        protected=True,
        position="execute",
        retry=True,
        user_message="Begin the task.",
    ),
    StageSpec(
        name="check_debug",
        agent_name="codechecker",
        protected=True,
        position="validate",
        retry=False,  # check_debug has its own internal loop
        user_message="Validate the best experiment config and its results.",
    ),
    StageSpec(
        name="publisher",
        agent_name="publisher",
        protected=False,
        position="publish",
        retry=False,
        optional=True,
        user_message="Publish the best model to HF Hub.",
    ),
]


def validate_pipeline(specs: list[StageSpec]) -> list[str]:
    """Validate a pipeline definition against safety rules.

    Returns a list of error messages (empty = valid).
    """
    errors: list[str] = []

    if not specs:
        errors.append("Pipeline must have at least one stage")
        return errors

    # Check for duplicate names
    names = [s.name for s in specs]
    seen: set[str] = set()
    for n in names:
        if n in seen:
            errors.append(f"Duplicate stage name: {n!r}")
        seen.add(n)

    # Check position values
    for s in specs:
        if s.position not in POSITION_ORDER:
            errors.append(
                f"Stage {s.name!r}: invalid position {s.position!r}. "
                f"Must be one of {POSITION_ORDER}"
            )

    # Check position ordering is respected
    for i in range(len(specs) - 1):
        pos_a = (
            POSITION_ORDER.index(specs[i].position)
            if specs[i].position in POSITION_ORDER else -1
        )
        pos_b = (
            POSITION_ORDER.index(specs[i + 1].position)
            if specs[i + 1].position in POSITION_ORDER else -1
        )
        if pos_a >= 0 and pos_b >= 0 and pos_a > pos_b:
            errors.append(
                f"Stage ordering violation: {specs[i].name!r} ({specs[i].position}) "
                f"cannot come before {specs[i + 1].name!r} ({specs[i + 1].position})"
            )

    # Check required position categories have at least one stage
    position_set = {s.position for s in specs}
    for required in ("plan", "execute", "validate"):
        if required not in position_set:
            errors.append(f"Pipeline must have at least one {required!r} stage")

    # Check all protected stages from DEFAULT_PIPELINE are present
    protected_names = {s.name for s in DEFAULT_PIPELINE if s.protected}
    present_names = {s.name for s in specs}
    missing_protected = protected_names - present_names
    if missing_protected:
        errors.append(
            f"Protected stages cannot be removed: {sorted(missing_protected)}"
        )

    # Check that protected stages are still marked protected
    for s in specs:
        default = _default_by_name(s.name)
        if default and default.protected and not s.protected:
            errors.append(
                f"Stage {s.name!r} is protected and cannot be made non-protected"
            )

    return errors


def _default_by_name(name: str) -> StageSpec | None:
    for s in DEFAULT_PIPELINE:
        if s.name == name:
            return s
    return None


def load_pipeline(path: Path | None = None) -> list[StageSpec]:
    """Load pipeline definition from JSON, falling back to DEFAULT_PIPELINE."""
    config_path = path or _CONFIG_PATH
    if not config_path.exists():
        return [StageSpec.from_dict(s.to_dict()) for s in DEFAULT_PIPELINE]
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        specs = [StageSpec.from_dict(d) for d in data]
        errors = validate_pipeline(specs)
        if errors:
            logger.warning(
                "Pipeline config has %d error(s), using defaults: %s",
                len(errors),
                "; ".join(errors),
            )
            return [StageSpec.from_dict(s.to_dict()) for s in DEFAULT_PIPELINE]
        return specs
    except Exception as exc:
        logger.warning("Failed to load pipeline config, using defaults: %s", exc)
        return [StageSpec.from_dict(s.to_dict()) for s in DEFAULT_PIPELINE]


def save_pipeline(specs: list[StageSpec], path: Path | None = None) -> None:
    """Persist pipeline definition to JSON after validation."""
    config_path = path or _CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps([s.to_dict() for s in specs], indent=2),
        encoding="utf-8",
    )
