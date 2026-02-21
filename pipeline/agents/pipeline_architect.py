"""PipelineArchitect agent — structural reasoning about pipeline composition
and meta-strategy adaptation.

This agent can inspect and modify the pipeline's stage sequence (adding,
removing, or reordering non-protected stages) and tune the orchestrator's
recovery strategy based on observed run history.
"""

from __future__ import annotations

from typing import Any, Callable

import pipeline.prompts.pipeline_architect_prompts  # noqa: F401
import pipeline.tools.agent_tools  # noqa: F401
import pipeline.tools.meta_strategy_tools  # noqa: F401
import pipeline.tools.orchestration_tools  # noqa: F401
from pipeline.agents.base import BaseAgentWrapper
from pipeline.prompts.fragments import assemble_prompt
from pipeline.tools.registry import build_toolset


class PipelineArchitectAgent(BaseAgentWrapper):
    """Modifies pipeline composition and meta-strategy with safety constraints."""

    agent_name = "pipeline_architect"

    def build_system_prompt(self, task: dict[str, Any]) -> str:
        return assemble_prompt(
            self.agent_name, task.get("channel", "default"), task
        )

    def build_tools(
        self, task: dict[str, Any]
    ) -> tuple[dict[str, Callable], list[dict]]:
        return build_toolset(
            # Pipeline composition
            "get_pipeline",
            "add_pipeline_stage",
            "remove_pipeline_stage",
            "reorder_pipeline_stage",
            # Meta-strategy
            "get_meta_strategy",
            "update_meta_strategy",
            "get_run_history",
            "analyze_strategy_effectiveness",
            "reset_meta_strategy",
            # Agent introspection
            "list_agents",
            "read_agent",
        )

    def build_context(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        context: list[dict[str, Any]] = []
        if task.get("run_history_summary"):
            context.append({
                "role": "user",
                "content": (
                    f"Recent run history summary:\n{task['run_history_summary']}"
                ),
            })
        if task.get("pipeline_issue"):
            context.append({
                "role": "user",
                "content": (
                    f"Pipeline issue to address:\n{task['pipeline_issue']}"
                ),
            })
        return context
