"""ComponentAuthor agent — writes new blocks/heads into open-synth-miner's registry."""

from __future__ import annotations

from typing import Any, Callable

import pipeline.prompts.author_prompts  # noqa: F401 — registers fragments
from pipeline.agents.base import BaseAgentWrapper
from pipeline.prompts.fragments import assemble_prompt
from pipeline.tools.registry import build_toolset


class ComponentAuthorAgent(BaseAgentWrapper):
    agent_name = "author"

    def build_system_prompt(self, task: dict[str, Any]) -> str:
        return assemble_prompt("author", task.get("channel", "default"), task)

    def build_tools(self, task: dict[str, Any]) -> tuple[dict[str, Callable], list[dict]]:
        tool_names = [
            # Discovery (read-only)
            "list_blocks",
            "list_heads",
            "list_component_files",
            "read_component",
            # Writing
            "write_component",
            "write_config",
            # Registry management
            "reload_registry",
        ]
        return build_toolset(*tool_names)

    def build_context(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        context = []
        if "component_spec" in task:
            context.append({
                "role": "user",
                "content": f"## Component Specification\n\n{task['component_spec']}",
            })
        if "reference_blocks" in task:
            context.append({
                "role": "user",
                "content": f"## Reference Blocks to Study\n\n{task['reference_blocks']}",
            })
        return context
