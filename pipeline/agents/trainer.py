"""Trainer agent — fits model parameters on historical data."""

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
            "read_file",
            "write_file",
            "get_historical_data",
            "compute_returns_stats",
            "check_shapes",
            "run_training_local",
            "run_python",
            "submit_basilica_job",
        ]
        return build_toolset(*tool_names)

    def build_context(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        context = []
        if "plan" in task:
            context.append({
                "role": "user",
                "content": f"## Planner Output\n\n```json\n{task['plan']}\n```",
            })
        if "model_path" in task:
            context.append({
                "role": "user",
                "content": f"The model to train/improve is at: `{task['model_path']}`",
            })
        return context
