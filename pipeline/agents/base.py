"""
BaseAgentWrapper — thin composition wrapper around SimpleAgent.

Each specialised agent (Planner, CodeChecker, Debugger, Trainer) subclasses
this and overrides ``build_context()`` to assemble agent-specific prompts,
tool sets, and context from prior pipeline stages.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from openai import OpenAI

from config import MAX_AGENT_TURNS, model_for
from pipeline.providers.chutes_client import get_chutes_client
from pipeline.providers.simple_agent import AgentResult, SimpleAgent

logger = logging.getLogger(__name__)


class BaseAgentWrapper:
    """Base class for all pipeline agents.

    Subclasses must implement:
        - ``agent_name``  (class attribute)
        - ``build_system_prompt(task)``
        - ``build_tools(task)``

    And may optionally override:
        - ``build_context(task)``  — prior messages to prepend
        - ``post_process(result)`` — transform the AgentResult
    """

    agent_name: str = "base"

    def __init__(
        self,
        client: OpenAI | None = None,
        model: str | None = None,
        max_turns: int | None = None,
        temperature: float = 0.1,
    ) -> None:
        self.client = client or get_chutes_client()
        self.model = model or model_for(self.agent_name)
        self.max_turns = max_turns or MAX_AGENT_TURNS
        self.temperature = temperature

    # ---------------------------------------------------------------- hooks
    def build_system_prompt(self, task: dict[str, Any]) -> str:
        """Return the system prompt for this agent given the task context."""
        raise NotImplementedError

    def build_tools(self, task: dict[str, Any]) -> tuple[dict[str, Callable], list[dict]]:
        """Return (tools_dict, tool_schemas) for this agent."""
        raise NotImplementedError

    def build_context(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        """Optional prior conversation context to prepend."""
        return []

    def post_process(self, result: AgentResult, task: dict[str, Any]) -> AgentResult:
        """Optional post-processing of the agent result."""
        return result

    # --------------------------------------------------------------- run
    def run(self, task: dict[str, Any]) -> AgentResult:
        """Build and execute the SimpleAgent for this task."""
        system_prompt = self.build_system_prompt(task)
        tools_dict, tool_schemas = self.build_tools(task)
        context = self.build_context(task)
        user_message = task.get("user_message", "Begin the task.")

        agent = SimpleAgent(
            client=self.client,
            model=self.model,
            system_prompt=system_prompt,
            tools=tools_dict,
            tool_schemas=tool_schemas,
            max_turns=self.max_turns,
            temperature=self.temperature,
        )

        logger.info(
            "Running %s (model=%s, tools=%s, temp=%.2f)",
            self.agent_name,
            self.model,
            list(tools_dict.keys()),
            self.temperature,
        )

        result = agent.run(user_message, context=context)
        result = self.post_process(result, task)

        logger.info(
            "%s finished: success=%s turns=%d",
            self.agent_name,
            result.success,
            result.turns_used,
        )
        return result
