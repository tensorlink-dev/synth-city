"""Trainer agent — executes experiments via ResearchSession."""

from __future__ import annotations

from typing import Any, Callable

import pipeline.prompts.trainer_prompts  # noqa: F401
from pipeline.agents.base import BaseAgentWrapper
from pipeline.prompts.fragments import assemble_prompt
from pipeline.tools.registry import build_toolset


class TrainerAgent(BaseAgentWrapper):
    agent_name = "trainer"

    def build_system_prompt(self, task: dict[str, Any]) -> str:
        return assemble_prompt("trainer", task.get("channel", "default"), task)

    def build_tools(self, task: dict[str, Any]) -> tuple[dict[str, Callable], list[dict]]:
        tool_names = [
            "create_experiment",
            "validate_experiment",
            "run_experiment",
            "run_preset",
            "sweep_presets",
            "compare_results",
            "session_summary",
        ]
        return build_toolset(*tool_names)

    def build_context(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        context = []
        if "plan" in task:
            context.append({
                "role": "user",
                "content": f"## Planner Output\n\n```json\n{task['plan']}\n```",
            })
        return context
