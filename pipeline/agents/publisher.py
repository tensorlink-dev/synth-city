"""Publisher agent — publishes best model to HF Hub with W&B tracking."""

from __future__ import annotations

from typing import Any, Callable

import pipeline.prompts.publisher_prompts  # noqa: F401
from pipeline.agents.base import BaseAgentWrapper
from pipeline.prompts.fragments import assemble_prompt
from pipeline.tools.registry import build_toolset


class PublisherAgent(BaseAgentWrapper):
    agent_name = "publisher"

    def build_system_prompt(self, task: dict[str, Any]) -> str:
        return assemble_prompt("publisher", task.get("channel", "default"), task)

    def build_tools(self, task: dict[str, Any]) -> tuple[dict[str, Callable], list[dict]]:
        tool_names = [
            "validate_experiment",
            "publish_model",
            "log_to_wandb",
        ]
        return build_toolset(*tool_names)

    def build_context(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        context = []
        if "best_experiment" in task:
            context.append({
                "role": "user",
                "content": f"## Best Experiment Config\n\n```json\n{task['best_experiment']}\n```",
            })
        if "best_metrics" in task:
            context.append({
                "role": "user",
                "content": f"## Best Metrics\n\n```json\n{task['best_metrics']}\n```",
            })
        if "comparison" in task:
            context.append({
                "role": "user",
                "content": f"## Full Comparison\n\n```json\n{task['comparison']}\n```",
            })
        return context
