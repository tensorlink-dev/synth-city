"""AgentDesigner agent — creates new pipeline agents, prompt modules, and tools."""

from __future__ import annotations

from typing import Any, Callable

import pipeline.prompts.agent_designer_prompts  # noqa: F401 — registers fragments
import pipeline.tools.tool_authoring  # noqa: F401 — registers tool authoring tools
from pipeline.agents.base import BaseAgentWrapper
from pipeline.prompts.fragments import assemble_prompt
from pipeline.tools.registry import build_toolset


class AgentDesignerAgent(BaseAgentWrapper):
    agent_name = "agent_designer"

    def build_system_prompt(self, task: dict[str, Any]) -> str:
        return assemble_prompt("agent_designer", task.get("channel", "default"), task)

    def build_tools(self, task: dict[str, Any]) -> tuple[dict[str, Callable], list[dict]]:
        tool_names = [
            # Agent discovery (read-only)
            "list_agents",
            "read_agent",
            "list_agent_prompts",
            "read_agent_prompt",
            "list_available_tools",
            # Agent writing
            "write_agent",
            "write_agent_prompt",
            # Tool authoring
            "write_tool",
            "reload_tools",
            "validate_tool",
            "read_tool",
            "list_tool_files",
            "describe_tool",
        ]
        return build_toolset(*tool_names)

    def build_context(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        context: list[dict[str, Any]] = []
        if "agent_spec" in task:
            context.append({
                "role": "user",
                "content": f"## Agent Specification\n\n{task['agent_spec']}",
            })
        if "reference_agents" in task:
            context.append({
                "role": "user",
                "content": f"## Reference Agents to Study\n\n{task['reference_agents']}",
            })
        return context
