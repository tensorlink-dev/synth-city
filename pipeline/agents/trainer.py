"""Trainer agent — executes experiments via ResearchSession."""

from __future__ import annotations

from typing import Any, Callable

import pipeline.prompts.trainer_prompts  # noqa: F401
import pipeline.tools.analysis_tools  # noqa: F401 — registers analysis tools
import pipeline.tools.data_loader  # noqa: F401 — registers data loader tools
import pipeline.tools.hippius_store  # noqa: F401 — registers hippius tools
import pipeline.tools.research_tools  # noqa: F401 — registers experiment tools
import pipeline.tools.training_tools  # noqa: F401 — registers training/GPU tools
from pipeline.agents.base import BaseAgentWrapper
from pipeline.prompts.fragments import assemble_prompt
from pipeline.tools.registry import build_toolset


class TrainerAgent(BaseAgentWrapper):
    agent_name = "trainer"

    def build_system_prompt(self, task: dict[str, Any]) -> str:
        return assemble_prompt("trainer", task.get("channel", "default"), task)

    def build_tools(self, task: dict[str, Any]) -> tuple[dict[str, Callable], list[dict]]:
        tool_names = [
            "create_experiment",
            "validate_experiment",
            "run_experiment",
            "run_preset",
            "sweep_presets",
            "compare_results",
            "session_summary",
            # Memory management
            "flush_session",
            # Data loading (HF OHLCV datasets — for local/lightweight use only)
            "create_data_loader",
            "data_loader_info",
            "split_data",
            # Historical analysis (persisted across restarts)
            "load_hippius_history",
            "fetch_experiment_runs",
            # Basilica GPU cloud — Docker-image-based deployments (only approach)
            "check_gpu_balance",
            "create_training_deployment",
            "get_training_deployment",
            "get_deployment_logs",
            "list_deployments",
            "delete_training_deployment",
            "wait_for_deployment_ready",
            "run_experiment_on_deployment",
        ]
        return build_toolset(*tool_names)

    def build_context(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        context = []
        if "plan" in task:
            context.append({
                "role": "user",
                "content": f"## Planner Output\n\n```json\n{task['plan']}\n```",
            })
        return context
