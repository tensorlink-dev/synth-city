"""
Meta-strategy tools — inspect and modify the pipeline's recovery strategy.

These tools give agents the ability to read, update, and analyze the
orchestrator's retry policy, temperature escalation, and stall detection
parameters based on observed run history.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from pipeline.meta_strategy import MetaStrategy, load_run_history
from pipeline.tools.registry import tool

logger = logging.getLogger(__name__)


@tool(
    description=(
        "Return the current meta-strategy configuration as JSON. "
        "Shows max_retries, temperature settings, stall detection, "
        "and any per-stage overrides."
    ),
)
def get_meta_strategy() -> str:
    """Return the current meta-strategy configuration."""
    try:
        strategy = MetaStrategy.load()
        return json.dumps(strategy.to_dict(), indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Update the meta-strategy configuration. "
        "Pass a JSON string of parameters to change. Only specified keys "
        "are updated; others keep their current values. "
        "All values are bounds-checked before applying. "
        "changes: JSON string, e.g. '{\"max_retries\": 8, \"temperature_step\": 0.15}'."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "changes": {
                "type": "string",
                "description": (
                    "JSON string of strategy parameters to update. "
                    "Valid keys: max_retries, base_temperature, temperature_step, "
                    "stall_threshold, cooldown_retries, per_stage_overrides."
                ),
            },
        },
        "required": ["changes"],
    },
)
def update_meta_strategy(changes: str) -> str:
    """Apply a partial update to the meta-strategy configuration."""
    try:
        try:
            updates: dict[str, Any] = json.loads(changes)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Invalid JSON: {exc}"})

        strategy = MetaStrategy.load()
        old = strategy.to_dict()

        # Apply updates
        known_fields = set(MetaStrategy.__dataclass_fields__.keys())
        unknown = set(updates.keys()) - known_fields
        if unknown:
            return json.dumps({
                "error": f"Unknown parameters: {sorted(unknown)}. "
                f"Valid: {sorted(known_fields)}",
            })

        for key, val in updates.items():
            setattr(strategy, key, val)

        # Validate
        errors = strategy.validate()
        if errors:
            return json.dumps({"error": "Validation failed", "details": errors})

        strategy.save()
        return json.dumps({
            "status": "updated",
            "previous": old,
            "current": strategy.to_dict(),
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Return recent pipeline run history from the JSONL log. "
        "last_n: number of recent events to return (default 100)."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "last_n": {
                "type": "integer",
                "description": "Number of recent events to return (default 100).",
            },
        },
        "required": [],
    },
)
def get_run_history(last_n: int = 100) -> str:
    """Return recent pipeline run events."""
    try:
        events = load_run_history(last_n=last_n)
        return json.dumps({
            "event_count": len(events),
            "events": events,
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Analyze meta-strategy effectiveness from run history. "
        "Computes summary statistics: average retries per stage, "
        "temperature at which success occurs, stall frequency, "
        "and success rates by stage."
    ),
)
def analyze_strategy_effectiveness() -> str:
    """Compute summary statistics from pipeline run history."""
    try:
        events = load_run_history(last_n=500)

        if not events:
            return json.dumps({"message": "No run history available yet."})

        # Per-stage stats
        stage_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "temperatures": [],
                "success_temperatures": [],
                "crps_values": [],
            }
        )

        # Per-pipeline stats
        pipeline_runs: dict[str, list[dict]] = defaultdict(list)

        for event in events:
            stage = event.get("stage", "unknown")
            stats = stage_stats[stage]
            stats["attempts"] += 1

            temp = event.get("temperature", 0.0)
            stats["temperatures"].append(temp)

            if event.get("success"):
                stats["successes"] += 1
                stats["success_temperatures"].append(temp)
            else:
                stats["failures"] += 1

            crps = event.get("crps")
            if crps is not None:
                stats["crps_values"].append(crps)

            run_id = event.get("pipeline_run_id", "unknown")
            pipeline_runs[run_id].append(event)

        # Compute summaries
        summary: dict[str, Any] = {
            "total_events": len(events),
            "unique_pipeline_runs": len(pipeline_runs),
            "stages": {},
        }

        for stage, stats in stage_stats.items():
            attempts = stats["attempts"]
            stage_summary: dict[str, Any] = {
                "total_attempts": attempts,
                "success_rate": (
                    round(stats["successes"] / attempts, 3) if attempts > 0 else 0.0
                ),
                "avg_temperature": (
                    round(sum(stats["temperatures"]) / len(stats["temperatures"]), 3)
                    if stats["temperatures"] else 0.0
                ),
            }
            if stats["success_temperatures"]:
                stage_summary["avg_success_temperature"] = round(
                    sum(stats["success_temperatures"]) / len(stats["success_temperatures"]),
                    3,
                )
            if stats["crps_values"]:
                stage_summary["best_crps"] = round(min(stats["crps_values"]), 4)
                stage_summary["avg_crps"] = round(
                    sum(stats["crps_values"]) / len(stats["crps_values"]), 4
                )
            summary["stages"][stage] = stage_summary

        # Overall success rate (pipeline level)
        pipeline_successes = 0
        for run_id, run_events in pipeline_runs.items():
            if any(e.get("success") for e in run_events):
                pipeline_successes += 1
        summary["pipeline_success_rate"] = round(
            pipeline_successes / len(pipeline_runs), 3
        ) if pipeline_runs else 0.0

        return json.dumps(summary, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description="Reset the meta-strategy to default values.",
)
def reset_meta_strategy() -> str:
    """Restore meta-strategy to factory defaults."""
    try:
        defaults = MetaStrategy()
        defaults.save()
        return json.dumps({
            "status": "reset",
            "strategy": defaults.to_dict(),
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
