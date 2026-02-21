"""
Orchestration tools — inspect and modify the pipeline stage sequence at runtime.

These tools give the PipelineArchitect agent the ability to add, remove, and
reorder pipeline stages while enforcing safety constraints (protected stages
cannot be removed, position ordering is preserved).
"""

from __future__ import annotations

import json
import logging

from pipeline.pipeline_def import (
    StageSpec,
    load_pipeline,
    save_pipeline,
    validate_pipeline,
)
from pipeline.tools.registry import tool

logger = logging.getLogger(__name__)


@tool(
    description=(
        "Return the current pipeline definition as JSON. "
        "Shows all stages with their names, agent names, positions, "
        "and whether they are protected."
    ),
)
def get_pipeline() -> str:
    """Return the current pipeline stage sequence."""
    try:
        specs = load_pipeline()
        return json.dumps({
            "stages": [s.to_dict() for s in specs],
            "stage_count": len(specs),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Add a new stage to the pipeline. "
        "name: unique stage identifier. "
        "agent_name: the agent to run (must be a registered agent). "
        "position: 'plan', 'execute', 'validate', 'publish', or 'post'. "
        "after_stage: name of the stage to insert after (optional, appends to "
        "position group if omitted). "
        "retry: whether to wrap with retry logic (default false). "
        "user_message: default message for the stage."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Unique stage name (e.g. 'evaluator').",
            },
            "agent_name": {
                "type": "string",
                "description": "Agent to run for this stage.",
            },
            "position": {
                "type": "string",
                "description": (
                    "Position category: 'plan', 'execute', 'validate', 'publish', or 'post'."
                ),
            },
            "after_stage": {
                "type": "string",
                "description": (
                    "Insert after this stage name. If omitted, appends at the "
                    "end of the position group."
                ),
            },
            "retry": {
                "type": "boolean",
                "description": "Whether to wrap with retry logic (default false).",
            },
            "user_message": {
                "type": "string",
                "description": "Default user message for the stage.",
            },
        },
        "required": ["name", "agent_name", "position"],
    },
)
def add_pipeline_stage(
    name: str,
    agent_name: str,
    position: str,
    after_stage: str = "",
    retry: bool = False,
    user_message: str = "Begin the task.",
) -> str:
    """Add a new stage to the pipeline."""
    try:
        specs = load_pipeline()

        # Check name uniqueness
        existing_names = {s.name for s in specs}
        if name in existing_names:
            return json.dumps({"error": f"Stage name already exists: {name!r}"})

        new_stage = StageSpec(
            name=name,
            agent_name=agent_name,
            protected=False,
            position=position,
            retry=retry,
            optional=True,
            user_message=user_message,
        )

        # Insert after specified stage, or append at end of position group
        if after_stage:
            idx = None
            for i, s in enumerate(specs):
                if s.name == after_stage:
                    idx = i + 1
                    break
            if idx is None:
                return json.dumps({
                    "error": f"after_stage not found: {after_stage!r}"
                })
            specs.insert(idx, new_stage)
        else:
            # Find last stage with same or earlier position, insert after it
            insert_at = len(specs)
            from pipeline.pipeline_def import POSITION_ORDER
            target_order = (
                POSITION_ORDER.index(position)
                if position in POSITION_ORDER
                else len(POSITION_ORDER)
            )
            for i in range(len(specs) - 1, -1, -1):
                pos_order = (
                    POSITION_ORDER.index(specs[i].position)
                    if specs[i].position in POSITION_ORDER
                    else len(POSITION_ORDER)
                )
                if pos_order <= target_order:
                    insert_at = i + 1
                    break
            specs.insert(insert_at, new_stage)

        # Validate
        errors = validate_pipeline(specs)
        if errors:
            return json.dumps({"error": "Validation failed", "details": errors})

        save_pipeline(specs)
        return json.dumps({
            "status": "added",
            "stage": new_stage.to_dict(),
            "pipeline_length": len(specs),
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Remove a non-protected stage from the pipeline. "
        "Protected stages (planner, trainer, check_debug) cannot be removed. "
        "name: the stage name to remove."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the stage to remove.",
            },
        },
        "required": ["name"],
    },
)
def remove_pipeline_stage(name: str) -> str:
    """Remove a non-protected stage from the pipeline."""
    try:
        specs = load_pipeline()

        target = None
        for s in specs:
            if s.name == name:
                target = s
                break

        if target is None:
            return json.dumps({"error": f"Stage not found: {name!r}"})

        if target.protected:
            return json.dumps({
                "error": f"Cannot remove protected stage: {name!r}"
            })

        specs = [s for s in specs if s.name != name]

        errors = validate_pipeline(specs)
        if errors:
            return json.dumps({"error": "Validation failed", "details": errors})

        save_pipeline(specs)
        return json.dumps({
            "status": "removed",
            "stage": name,
            "pipeline_length": len(specs),
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Move a non-protected stage to a new position in the pipeline. "
        "name: the stage to move. "
        "after_stage: place it after this stage."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the stage to move.",
            },
            "after_stage": {
                "type": "string",
                "description": "Place after this stage name.",
            },
        },
        "required": ["name", "after_stage"],
    },
)
def reorder_pipeline_stage(name: str, after_stage: str) -> str:
    """Move a non-protected stage to a new position."""
    try:
        specs = load_pipeline()

        target = None
        for s in specs:
            if s.name == name:
                target = s
                break

        if target is None:
            return json.dumps({"error": f"Stage not found: {name!r}"})

        if target.protected:
            return json.dumps({
                "error": f"Cannot reorder protected stage: {name!r}"
            })

        if name == after_stage:
            return json.dumps({
                "error": "Cannot place a stage after itself"
            })

        # Find insertion point
        insert_after_idx = None
        for i, s in enumerate(specs):
            if s.name == after_stage:
                insert_after_idx = i
                break
        if insert_after_idx is None:
            return json.dumps({
                "error": f"after_stage not found: {after_stage!r}"
            })

        # Remove and reinsert
        specs = [s for s in specs if s.name != name]
        # Recalculate index after removal
        for i, s in enumerate(specs):
            if s.name == after_stage:
                insert_after_idx = i
                break
        specs.insert(insert_after_idx + 1, target)

        errors = validate_pipeline(specs)
        if errors:
            return json.dumps({"error": "Validation failed", "details": errors})

        save_pipeline(specs)
        return json.dumps({
            "status": "reordered",
            "stage": name,
            "after": after_stage,
            "pipeline": [s.name for s in specs],
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
