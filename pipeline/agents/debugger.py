"""Debugger agent — fixes code that failed CodeChecker validation."""

from __future__ import annotations

from typing import Any, Callable

import pipeline.prompts.debugger_prompts  # noqa: F401
from pipeline.agents.base import BaseAgentWrapper
from pipeline.prompts.fragments import assemble_prompt
from pipeline.tools.registry import build_toolset


class DebuggerAgent(BaseAgentWrapper):
    agent_name = "debugger"

    def build_system_prompt(self, task: dict[str, Any]) -> str:
        return assemble_prompt("debugger", task.get("channel", "default"), task)

    def build_tools(self, task: dict[str, Any]) -> tuple[dict[str, Callable], list[dict]]:
        tool_names = ["read_file", "write_file", "check_shapes", "run_python"]
        return build_toolset(*tool_names)

    def build_context(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        context = []
        # Inject the error report from CodeChecker
        if "error_report" in task:
            context.append({
                "role": "user",
                "content": f"## CodeChecker Error Report\n\n{task['error_report']}",
            })
        if "model_path" in task:
            context.append({
                "role": "user",
                "content": f"The model to fix is at: `{task['model_path']}`",
            })
        return context
