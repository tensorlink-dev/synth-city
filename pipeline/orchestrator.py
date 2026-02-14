"""
Pipeline orchestrator — conditional retry loops with temperature escalation,
stall detection, adaptive authoring, and dynamic context injection.

This is the top-level controller that chains agents together:
    Planner → [Author] → Trainer → [Author → re-Train] → Check/Debug → Publisher

Agents have autonomy at every stage:
    - The Planner can request new components via ``author_requests`` in its plan.
    - The Trainer can author new blocks inline when existing ones are insufficient.
    - The Debugger can author replacement blocks as a last resort.
    - The orchestrator runs an adaptive Author loop when training CRPS is poor.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pipeline.agents.author import ComponentAuthorAgent
from pipeline.agents.code_checker import CodeCheckerAgent
from pipeline.agents.debugger import DebuggerAgent
from pipeline.agents.planner import PlannerAgent
from pipeline.agents.publisher import PublisherAgent
from pipeline.agents.trainer import TrainerAgent
from pipeline.providers.simple_agent import AgentResult

logger = logging.getLogger(__name__)

# CRPS ratio above the historical best that triggers an authoring loop.
_AUTHOR_CRPS_RATIO = 2.0


class PipelineOrchestrator:
    """Orchestrates the full pipeline with adaptive agent autonomy.

    Key features:
        - **Retry with escalation**: failed steps retry with increasing
          temperature (0.1 → 0.2 → 0.3 …).
        - **Stall detection**: if a Debugger produces the same experiment config
          as the previous attempt, inject a CRITICAL WARNING.
        - **Adaptive authoring**: if the Planner requests new components or
          the Trainer can't beat a CRPS threshold, the Author agent is invoked
          to create custom blocks, then training reruns with expanded search space.
        - **Inline authoring**: Trainer and Debugger agents have authoring tools
          so they can create new blocks on the fly without waiting for the
          orchestrator.
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
        """Execute the full pipeline with adaptive authoring.

        Parameters
        ----------
        task : dict
            Optional keys:
                - ``crps_scores``: recent CRPS scores JSON
                - ``channel``: prompt channel selector
                - ``prior_comparison``: prior experiment comparison

        Flow
        ----
        1. Planner — produce experiment plan (may include ``author_requests``)
        2. Author  — if Planner requested new components, build them first
        3. Trainer  — execute experiments (has inline authoring tools)
        4. Author + re-Train — if CRPS is poor, invoke Author then re-train
        5. Check/Debug — validate best experiment (Debugger has authoring tools)
        6. Publisher — optional
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
        task["plan"] = (
            json.dumps(plan_data, indent=2) if isinstance(plan_data, dict)
            else str(plan_data)
        )

        # Stage 1b: Author (if Planner requested new components)
        author_requests = self._extract_author_requests(plan_data)
        if author_requests:
            logger.info("=== STAGE 1b: PLANNER-REQUESTED AUTHORING (%d) ===", len(author_requests))
            for req in author_requests:
                author_result = self._run_author(task, req)
                results["stages"].append({
                    "agent": "author",
                    "trigger": "planner_request",
                    "request": req.get("name", "unknown"),
                    "success": author_result.success,
                })
                if author_result.success:
                    logger.info("Author created component: %s", req.get("name"))
                else:
                    logger.warning("Author failed for: %s", req.get("name"))

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

        # Stage 2b: Adaptive authoring loop — if CRPS is poor, invoke Author
        # then re-train with expanded component library
        if best and self._should_author_adaptively(best):
            logger.info("=== STAGE 2b: ADAPTIVE AUTHORING (CRPS too high) ===")
            adaptive_result = self._adaptive_author_loop(task, best)
            results["stages"].append(adaptive_result)
            # Update best if the re-train produced better results
            if adaptive_result.get("improved"):
                best = adaptive_result.get("best") or best
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

        # Persist pipeline summary to Hippius
        self._save_to_hippius(results)

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

    # ----------------------------------------------------------- authoring
    @staticmethod
    def _extract_author_requests(plan_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Pull ``author_requests`` from the Planner's structured output."""
        if not isinstance(plan_data, dict):
            return []
        requests = plan_data.get("author_requests", [])
        if isinstance(requests, list):
            return [r for r in requests if isinstance(r, dict) and r.get("name")]
        return []

    def _run_author(
        self, task: dict[str, Any], request: dict[str, Any],
    ) -> AgentResult:
        """Run the ComponentAuthor agent for a single authoring request."""
        author_task = dict(task)
        author_task["component_spec"] = json.dumps(request, indent=2)
        refs = request.get("reference_blocks", [])
        if refs:
            author_task["reference_blocks"] = ", ".join(refs)
        author_task["user_message"] = (
            f"Create a new {request.get('type', 'block')} called "
            f"\"{request['name']}\". Rationale: {request.get('rationale', 'improve CRPS')}. "
            f"Study existing components first, then write and register the new component."
        )
        agent = ComponentAuthorAgent(temperature=self.base_temperature)
        return agent.run(author_task)

    def _should_author_adaptively(self, best: dict[str, Any]) -> bool:
        """Decide whether to trigger the adaptive authoring loop.

        Returns True when the best CRPS from training is poor enough that
        authoring new components might help.  Uses a simple heuristic:
        if no CRPS was produced at all, or CRPS is missing, skip (nothing
        to improve on).
        """
        crps = best.get("crps")
        if crps is None:
            return False
        try:
            crps_val = float(crps)
        except (TypeError, ValueError):
            return False
        # Trigger if CRPS > 1.0 (a generous threshold — most competitive
        # models are well below 1.0).  This avoids wasting an authoring loop
        # when training already found a good architecture.
        return crps_val > 1.0

    def _adaptive_author_loop(
        self, task: dict[str, Any], best: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke Author with diagnostic context, then re-run Trainer.

        This gives the pipeline a second chance: if the first round of training
        only produced mediocre CRPS, the Author creates a targeted block based
        on what was tried and what failed, then the Trainer reruns with the
        expanded component library.
        """
        crps = best.get("crps", "unknown")
        experiment = best.get("experiment", "")

        # Build a diagnostic brief for the Author
        author_task = dict(task)
        author_task["component_spec"] = json.dumps({
            "trigger": "adaptive",
            "current_best_crps": crps,
            "current_best_experiment": experiment,
            "goal": (
                "Create a new backbone block that addresses weaknesses in the "
                "current best architecture. Focus on improving CRPS."
            ),
        }, indent=2)
        author_task["user_message"] = (
            f"The best CRPS so far is {crps}, which is not competitive. "
            f"The best experiment was: {experiment[:500]}\n\n"
            f"Study the existing blocks, identify what's missing, and write a new "
            f"block that could improve calibration or capture patterns the current "
            f"blocks miss. Then reload the registry."
        )

        logger.info("Running Author agent for adaptive component creation")
        author = ComponentAuthorAgent(temperature=self.base_temperature + self.temperature_step)
        author_result = author.run(author_task)

        stage_result: dict[str, Any] = {
            "agent": "author",
            "trigger": "adaptive_crps",
            "author_success": author_result.success,
            "improved": False,
        }

        if not author_result.success:
            logger.warning("Adaptive Author failed — skipping re-train")
            return stage_result

        # Re-run Trainer with expanded component library
        logger.info("Re-running Trainer with expanded component library")
        task["user_message"] = (
            "New components have been authored and registered. Re-run experiments "
            "using the expanded component library. Call list_blocks() to discover "
            "the new blocks, then create and run experiments with them."
        )
        retrain_result = self._run_with_retry(
            TrainerAgent, task, stage_name="trainer_retrain",
        )
        stage_result["retrain_success"] = retrain_result.success

        if retrain_result.success:
            new_best = self._extract_best(retrain_result)
            if new_best:
                new_crps = new_best.get("crps")
                old_crps = best.get("crps")
                try:
                    improved = (
                        new_crps is not None
                        and old_crps is not None
                        and float(new_crps) < float(old_crps)
                    )
                except (TypeError, ValueError):
                    improved = False

                if improved:
                    logger.info(
                        "Adaptive authoring improved CRPS: %s → %s",
                        old_crps, new_crps,
                    )
                    stage_result["improved"] = True
                    stage_result["best"] = new_best
                    stage_result["old_crps"] = old_crps
                    stage_result["new_crps"] = new_crps

        return stage_result

    # ----------------------------------------------------------- publisher
    def _run_publisher(self, task: dict[str, Any]) -> AgentResult:
        agent = PublisherAgent(temperature=self.base_temperature)
        task["user_message"] = "Publish the best model to HF Hub."
        return agent.run(task)

    # ----------------------------------------------------------- persistence
    @staticmethod
    def _save_to_hippius(results: dict[str, Any]) -> None:
        """Best-effort save of pipeline results to Hippius storage."""
        try:
            from pipeline.tools.hippius_store import (
                reset_run_id,
                save_comparison,
                save_pipeline_summary,
            )

            save_pipeline_summary(results)

            # Also persist the comparison snapshot if available
            best_exp = results.get("best_experiment")
            if best_exp:
                save_comparison({
                    "best_experiment": best_exp,
                    "best_crps": results.get("best_crps"),
                    "success": results.get("success", False),
                })

            # Reset run ID so next pipeline invocation gets a fresh one
            reset_run_id()
        except Exception as exc:
            logger.debug("Hippius save skipped: %s", exc)

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
