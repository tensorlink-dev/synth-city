"""Planner agent — discovers components and produces an experiment plan."""

from __future__ import annotations

from typing import Any, Callable

import pipeline.prompts.planner_prompts  # noqa: F401 — registers fragments
import pipeline.tools.analysis_tools  # noqa: F401 — registers analysis tools
import pipeline.tools.hippius_store  # noqa: F401 — registers hippius tools
import pipeline.tools.research_tools  # noqa: F401 — registers experiment tools
from pipeline.agents.base import BaseAgentWrapper
from pipeline.prompts.fragments import assemble_prompt
from pipeline.tools.registry import build_toolset


class PlannerAgent(BaseAgentWrapper):
    agent_name = "planner"

    def build_system_prompt(self, task: dict[str, Any]) -> str:
        return assemble_prompt("planner", task.get("channel", "default"), task)

    def build_tools(self, task: dict[str, Any]) -> tuple[dict[str, Callable], list[dict]]:
        tool_names = [
            "list_blocks",
            "list_heads",
            "list_presets",
            "session_summary",
            "compare_results",
            # Historical analysis
            "load_hippius_history",
            "load_hippius_run",
            "fetch_wandb_runs",
            "analyze_wandb_trends",
            "list_hf_models",
        ]
        return build_toolset(*tool_names)

    def build_context(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        context = []
        if "crps_scores" in task:
            context.append({
                "role": "user",
                "content": f"## Recent CRPS Scores\n\n```json\n{task['crps_scores']}\n```",
            })
        if "prior_comparison" in task:
            context.append({
                "role": "user",
                "content": (
                    "## Prior Experiment Comparison\n\n"
                    f"```json\n{task['prior_comparison']}\n```"
                ),
            })
        return context
