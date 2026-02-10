"""CodeChecker agent — validates model code against SN50 requirements."""

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
        tool_names = ["read_file", "check_shapes", "run_python"]
        return build_toolset(*tool_names)

    def build_context(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        context = []
        model_path = task.get("model_path")
        if model_path:
            context.append({
                "role": "user",
                "content": f"Validate the model at: `{model_path}`",
            })
        return context
