"""CodeChecker agent — validates experiment configs via ResearchSession."""

from __future__ import annotations

from typing import Any, Callable

import pipeline.prompts.checker_prompts  # noqa: F401
from pipeline.agents.base import BaseAgentWrapper
from pipeline.prompts.fragments import assemble_prompt
from pipeline.tools.registry import build_toolset


class CodeCheckerAgent(BaseAgentWrapper):
    agent_name = "codechecker"

    def build_system_prompt(self, task: dict[str, Any]) -> str:
        return assemble_prompt("codechecker", task.get("channel", "default"), task)

    def build_tools(self, task: dict[str, Any]) -> tuple[dict[str, Callable], list[dict]]:
        tool_names = [
            "validate_experiment",
            "describe_experiment",
            "list_blocks",
            "list_heads",
        ]
        return build_toolset(*tool_names)

    def build_context(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        context = []
        if "experiment" in task:
            context.append({
                "role": "user",
                "content": f"## Experiment Config to Validate\n\n```json\n{task['experiment']}\n```",
            })
        if "run_result" in task:
            context.append({
                "role": "user",
                "content": f"## Experiment Run Result\n\n```json\n{task['run_result']}\n```",
            })
        return context
