"""Planner agent — analyses current state and produces an improvement plan."""

from __future__ import annotations

from typing import Any, Callable

import pipeline.prompts.planner_prompts  # noqa: F401 — registers fragments
from config import SN50_ASSETS
from pipeline.agents.base import BaseAgentWrapper
from pipeline.prompts.fragments import assemble_prompt
from pipeline.tools.registry import build_toolset


class PlannerAgent(BaseAgentWrapper):
    agent_name = "planner"

    def build_system_prompt(self, task: dict[str, Any]) -> str:
        task.setdefault("assets", ", ".join(f"{k} (w={v})" for k, v in SN50_ASSETS.items()))
        return assemble_prompt("planner", task.get("channel", "default"), task)

    def build_tools(self, task: dict[str, Any]) -> tuple[dict[str, Callable], list[dict]]:
        tool_names = [
            "read_file",
            "write_file",
            "get_historical_data",
            "compute_returns_stats",
            "check_shapes",
        ]
        return build_toolset(*tool_names)

    def build_context(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        context = []
        # Inject prior CRPS scores if available
        if "crps_scores" in task:
            context.append({
                "role": "user",
                "content": f"## Recent CRPS Scores\n\n```json\n{task['crps_scores']}\n```",
            })
        # Inject prior model path if available
        if "current_model_path" in task:
            context.append({
                "role": "user",
                "content": f"The current model code is at: `{task['current_model_path']}`",
            })
        return context
