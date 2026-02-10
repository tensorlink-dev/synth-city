"""Debugger agent — fixes experiment configs or execution failures."""

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
        tool_names = [
            "create_experiment",
            "validate_experiment",
            "run_experiment",
            "list_blocks",
            "list_heads",
        ]
        return build_toolset(*tool_names)

    def build_context(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        context = []
        if "error_report" in task:
            context.append({
                "role": "user",
                "content": f"## Error Report\n\n{task['error_report']}",
            })
        if "experiment" in task:
            context.append({
                "role": "user",
                "content": f"## Failed Experiment Config\n\n```json\n{task['experiment']}\n```",
            })
        return context
