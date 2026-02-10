"""
Pipeline orchestrator — conditional retry loops with temperature escalation,
stall detection, and dynamic context injection.

This is the top-level controller that chains agents together:
    Planner → Trainer → CodeChecker → (Debugger if failed) → repeat
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from config import WORKSPACE_DIR
from pipeline.agents.code_checker import CodeCheckerAgent
from pipeline.agents.debugger import DebuggerAgent
from pipeline.agents.planner import PlannerAgent
from pipeline.agents.trainer import TrainerAgent
from pipeline.providers.simple_agent import AgentResult

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the full Planner → Train → Check → Debug cycle.

    Key features:
        - **Retry with escalation**: failed steps retry with increasing
          temperature (0.1 → 0.2 → 0.3 …).
        - **Stall detection**: if the Debugger writes the same file content
          twice in a row, inject a CRITICAL WARNING.
        - **Ephemeral compression**: large context from earlier agents is
          summarised before being passed to later agents.
    """

    def __init__(
        self,
        max_retries: int = 5,
        base_temperature: float = 0.1,
        temperature_step: float = 0.1,
    ) -> None:
        self.max_retries = max_retries
        self.base_temperature = base_temperature
        self.temperature_step = temperature_step

    # ---------------------------------------------------------------- main
    def run(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute the full pipeline.

        Parameters
        ----------
        task : dict
            Must contain at least:
                - ``target_assets``: list of asset symbols to target
                - ``current_model_path``: path to the current model file (optional)
            Optional:
                - ``crps_scores``: recent CRPS scores JSON
                - ``channel``: prompt channel selector
        """
        task.setdefault("channel", "default")
        task.setdefault("target_assets", ["BTC", "ETH", "SOL"])

        results: dict[str, Any] = {"stages": [], "success": False}

        # Stage 1: Plan
        logger.info("=== STAGE 1: PLANNER ===")
        plan_result = self._run_planner(task)
        results["stages"].append({"agent": "planner", "success": plan_result.success})

        if not plan_result.success:
            logger.error("Planner failed — aborting pipeline")
            return results

        # Extract plan for downstream agents
        plan_data = plan_result.structured or {}
        task["plan"] = json.dumps(plan_data, indent=2) if isinstance(plan_data, dict) else str(plan_data)

        # Stage 2: Train
        logger.info("=== STAGE 2: TRAINER ===")
        train_result = self._run_with_retry(
            TrainerAgent,
            task,
            stage_name="trainer",
        )
        results["stages"].append({"agent": "trainer", "success": train_result.success})

        if not train_result.success:
            logger.error("Trainer failed after retries — aborting pipeline")
            return results

        # Extract model path from trainer result
        model_path = self._extract_model_path(train_result)
        if model_path:
            task["model_path"] = model_path

        # Stage 3: Check → Debug loop
        logger.info("=== STAGE 3: CHECK/DEBUG LOOP ===")
        check_debug_result = self._check_debug_loop(task)
        results["stages"].append(check_debug_result)
        results["success"] = check_debug_result.get("passed", False)
        results["model_path"] = task.get("model_path")

        return results

    # ----------------------------------------------------------- planner
    def _run_planner(self, task: dict[str, Any]) -> AgentResult:
        agent = PlannerAgent(temperature=self.base_temperature)
        task["user_message"] = (
            f"Analyse the current model and market conditions for assets: "
            f"{', '.join(task['target_assets'])}. "
            f"Produce an improvement plan."
        )
        return agent.run(task)

    # ----------------------------------------------------------- retry wrapper
    def _run_with_retry(
        self,
        agent_cls: type,
        task: dict[str, Any],
        stage_name: str,
    ) -> AgentResult:
        """Run an agent with escalating temperature on failure."""
        last_result: AgentResult | None = None

        for attempt in range(self.max_retries):
            temp = self.base_temperature + (attempt * self.temperature_step)
            logger.info(
                "%s attempt %d/%d (temp=%.2f)",
                stage_name,
                attempt + 1,
                self.max_retries,
                temp,
            )

            agent = agent_cls(temperature=temp)

            # Inject failure context from previous attempt
            if last_result and not last_result.success:
                task["user_message"] = (
                    f"Previous attempt failed. Error: {last_result.raw_text[:1000]}\n\n"
                    f"Please try a different approach. Attempt {attempt + 1}/{self.max_retries}."
                )
            elif "user_message" not in task:
                task["user_message"] = "Begin the task."

            result = agent.run(task)
            if result.success:
                return result

            last_result = result
            logger.warning("%s attempt %d failed", stage_name, attempt + 1)

        return last_result or AgentResult(success=False, raw_text="No attempts made")

    # ----------------------------------------------------------- check/debug
    def _check_debug_loop(self, task: dict[str, Any]) -> dict[str, Any]:
        """Alternating CodeChecker → Debugger loop with stall detection."""
        file_hashes: list[str] = []
        stage_results: list[dict] = []

        for attempt in range(self.max_retries):
            # Check
            temp = self.base_temperature + (attempt * self.temperature_step)
            checker = CodeCheckerAgent(temperature=self.base_temperature)
            task["user_message"] = f"Validate the model at: `{task.get('model_path', 'model.py')}`"
            check_result = checker.run(task)

            stage_results.append({
                "agent": "codechecker",
                "attempt": attempt,
                "success": check_result.success,
            })

            if check_result.success:
                # Check if the structured result indicates pass
                structured = check_result.structured or {}
                passed = structured.get("success", True) if isinstance(structured, dict) else True
                if passed:
                    return {"passed": True, "attempts": attempt + 1, "stages": stage_results}

            # Debug
            logger.info("CodeChecker failed — running Debugger (attempt %d)", attempt + 1)
            task["error_report"] = check_result.raw_text[:3000]

            # Stall detection: hash the model file
            model_path = task.get("model_path", "model.py")
            current_hash = self._hash_file(model_path)
            if current_hash and current_hash in file_hashes:
                # STALL DETECTED — inject warning
                logger.warning("STALL DETECTED: Debugger did not modify the file")
                task["user_message"] = (
                    "⚠️ CRITICAL WARNING: The model file has NOT changed since the last "
                    "attempt. You MUST take a DIFFERENT approach. Re-read the error "
                    "carefully and try a fundamentally different fix strategy.\n\n"
                    f"Error report:\n{task['error_report']}"
                )
            else:
                task["user_message"] = f"Fix the model. Error report:\n{task['error_report']}"

            if current_hash:
                file_hashes.append(current_hash)

            debugger = DebuggerAgent(temperature=temp)
            debug_result = debugger.run(task)
            stage_results.append({
                "agent": "debugger",
                "attempt": attempt,
                "success": debug_result.success,
            })

        return {"passed": False, "attempts": self.max_retries, "stages": stage_results}

    # ----------------------------------------------------------- helpers
    @staticmethod
    def _extract_model_path(result: AgentResult) -> str | None:
        """Try to extract the model file path from a trainer result."""
        if isinstance(result.structured, dict):
            for key in ("model_path", "path", "output_path", "file"):
                if key in result.structured:
                    return result.structured[key]
            # Check nested "result" field
            nested = result.structured.get("result", {})
            if isinstance(nested, dict):
                for key in ("model_path", "path", "output_path", "file"):
                    if key in nested:
                        return nested[key]
        return None

    @staticmethod
    def _hash_file(path: str) -> str | None:
        """Hash a workspace file for stall detection."""
        full_path = WORKSPACE_DIR / path
        if full_path.exists():
            content = full_path.read_bytes()
            return hashlib.sha256(content).hexdigest()
        return None
