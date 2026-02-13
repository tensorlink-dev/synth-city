"""
Pipeline orchestrator — conditional retry loops with temperature escalation,
stall detection, and dynamic context injection.

This is the top-level controller that chains agents together:
    Planner → Trainer → CodeChecker → (Debugger if failed) → Publisher
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pipeline.agents.code_checker import CodeCheckerAgent
from pipeline.agents.debugger import DebuggerAgent
from pipeline.agents.planner import PlannerAgent
from pipeline.agents.publisher import PublisherAgent
from pipeline.agents.trainer import TrainerAgent
from pipeline.providers.simple_agent import AgentResult

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the full Planner → Train → Check → Debug → Publish cycle.

    Key features:
        - **Retry with escalation**: failed steps retry with increasing
          temperature (0.1 → 0.2 → 0.3 …).
        - **Stall detection**: if a Debugger produces the same experiment config
          as the previous attempt, inject a CRITICAL WARNING.
        - **Ephemeral compression**: large context from earlier agents is
          summarised before being passed to later agents.
    """

    def __init__(
        self,
        max_retries: int = 5,
        base_temperature: float = 0.1,
        temperature_step: float = 0.1,
        publish: bool = False,
    ) -> None:
        self.max_retries = max_retries
        self.base_temperature = base_temperature
        self.temperature_step = temperature_step
        self.publish = publish

    # ---------------------------------------------------------------- main
    def run(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute the full pipeline.

        Parameters
        ----------
        task : dict
            Optional keys:
                - ``crps_scores``: recent CRPS scores JSON
                - ``channel``: prompt channel selector
                - ``prior_comparison``: prior experiment comparison
        """
        task.setdefault("channel", "default")
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

        # Stage 2: Train (execute experiments from the plan)
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

        # Extract best experiment and metrics from trainer result
        best = self._extract_best(train_result)
        if best:
            task["best_experiment"] = best.get("experiment", "")
            task["best_metrics"] = best.get("metrics", "")
            task["comparison"] = best.get("comparison", "")
            task["experiment"] = best.get("experiment", "")

        # Stage 3: Check → Debug loop on the best experiment
        logger.info("=== STAGE 3: CHECK/DEBUG LOOP ===")
        check_debug_result = self._check_debug_loop(task)
        results["stages"].append(check_debug_result)
        results["success"] = check_debug_result.get("passed", False)

        # Stage 4: Publish (optional)
        if self.publish and results["success"]:
            logger.info("=== STAGE 4: PUBLISHER ===")
            pub_result = self._run_publisher(task)
            results["stages"].append({"agent": "publisher", "success": pub_result.success})
            results["published"] = pub_result.success
            if pub_result.structured:
                results["publish_info"] = pub_result.structured

        # Attach best experiment info to results
        if best:
            results["best_experiment"] = best.get("experiment")
            results["best_crps"] = best.get("crps")

        return results

    # ----------------------------------------------------------- planner
    def _run_planner(self, task: dict[str, Any]) -> AgentResult:
        agent = PlannerAgent(temperature=self.base_temperature)
        task["user_message"] = (
            "Discover the available components and produce an experiment plan "
            "to find the best architecture for SN50 CRPS."
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
        prev_experiment_json: str | None = None
        stage_results: list[dict] = []

        for attempt in range(self.max_retries):
            # Check
            checker = CodeCheckerAgent(temperature=self.base_temperature)
            task["user_message"] = "Validate the best experiment config and its results."
            check_result = checker.run(task)

            stage_results.append({
                "agent": "codechecker",
                "attempt": attempt,
                "success": check_result.success,
            })

            if check_result.success:
                structured = check_result.structured or {}
                passed = structured.get("success", True) if isinstance(structured, dict) else True
                if passed:
                    return {"passed": True, "attempts": attempt + 1, "stages": stage_results}

            # Debug
            temp = self.base_temperature + (attempt * self.temperature_step)
            logger.info("CodeChecker failed — running Debugger (attempt %d)", attempt + 1)
            task["error_report"] = check_result.raw_text[:3000]

            # Stall detection: compare experiment JSON across attempts
            current_exp = task.get("best_experiment", "")
            if current_exp and current_exp == prev_experiment_json:
                logger.warning("STALL DETECTED: experiment config unchanged")
                task["user_message"] = (
                    "CRITICAL WARNING: The experiment config has NOT changed since the last "
                    "attempt. You MUST take a DIFFERENT approach — change blocks, head, "
                    "d_model, or learning rate.\n\n"
                    f"Error report:\n{task['error_report']}"
                )
            else:
                task["user_message"] = f"Fix the experiment. Error report:\n{task['error_report']}"
            prev_experiment_json = current_exp

            debugger = DebuggerAgent(temperature=temp)
            debug_result = debugger.run(task)
            stage_results.append({
                "agent": "debugger",
                "attempt": attempt,
                "success": debug_result.success,
            })

            # Update experiment from debugger output
            if debug_result.success and debug_result.structured:
                fixed = debug_result.structured
                if isinstance(fixed, dict):
                    for key in ("experiment", "result", "config"):
                        if key in fixed:
                            task["best_experiment"] = (
                                json.dumps(fixed[key]) if isinstance(fixed[key], dict)
                                else str(fixed[key])
                            )
                            task["experiment"] = task["best_experiment"]
                            break

        return {"passed": False, "attempts": self.max_retries, "stages": stage_results}

    # ----------------------------------------------------------- publisher
    def _run_publisher(self, task: dict[str, Any]) -> AgentResult:
        agent = PublisherAgent(temperature=self.base_temperature)
        task["user_message"] = "Publish the best model to HF Hub."
        return agent.run(task)

    # ----------------------------------------------------------- helpers
    @staticmethod
    def _extract_best(result: AgentResult) -> dict[str, Any] | None:
        """Extract the best experiment info from a trainer result."""
        if not isinstance(result.structured, dict):
            return None

        structured = result.structured
        best: dict[str, Any] = {}

        # Direct keys
        for key in ("best_experiment", "experiment", "config"):
            if key in structured:
                val = structured[key]
                best["experiment"] = json.dumps(val) if isinstance(val, dict) else str(val)
                break

        # Nested result
        nested = structured.get("result", structured)
        if isinstance(nested, dict):
            for key in ("best_experiment", "experiment", "config"):
                if key in nested:
                    val = nested[key]
                    best["experiment"] = json.dumps(val) if isinstance(val, dict) else str(val)
                    break

            if "crps" in nested:
                best["crps"] = nested["crps"]
            elif "metrics" in nested and isinstance(nested["metrics"], dict):
                best["crps"] = nested["metrics"].get("crps")
                best["metrics"] = json.dumps(nested["metrics"])

            if "comparison" in nested:
                best["comparison"] = (
                    json.dumps(nested["comparison"]) if isinstance(nested["comparison"], dict)
                    else str(nested["comparison"])
                )

        return best if best else None
