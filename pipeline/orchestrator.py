"""
Pipeline orchestrator — data-driven stage sequencing with adaptive meta-strategy,
retry loops with temperature escalation, stall detection, and dynamic context
injection.

This is the top-level controller that chains agents together.  The pipeline
stage sequence is defined declaratively via ``StageSpec`` objects (see
``pipeline_def.py``) and can be modified at runtime by the PipelineArchitect
agent.  Recovery parameters (retry counts, temperature, stall detection) are
managed by ``MetaStrategy`` and can be tuned based on observed run history.
"""

from __future__ import annotations

import importlib
import json
import logging
import uuid
from typing import Any

from pipeline.agents.code_checker import CodeCheckerAgent
from pipeline.agents.debugger import DebuggerAgent
from pipeline.agents.planner import PlannerAgent
from pipeline.agents.publisher import PublisherAgent
from pipeline.agents.trainer import TrainerAgent
from pipeline.meta_strategy import MetaStrategy, record_run_event
from pipeline.monitor import get_monitor
from pipeline.pipeline_def import StageSpec, load_pipeline
from pipeline.providers.simple_agent import AgentResult

logger = logging.getLogger(__name__)
_mon = get_monitor()

# Core agents — always available without dynamic discovery
_CORE_AGENTS: dict[str, type] = {
    "planner": PlannerAgent,
    "trainer": TrainerAgent,
    "codechecker": CodeCheckerAgent,
    "debugger": DebuggerAgent,
    "publisher": PublisherAgent,
}


def resolve_agent(agent_name: str) -> type | None:
    """Resolve an agent class by name.

    Checks the core agent dict first, then falls back to dynamic import
    from ``pipeline.agents.<agent_name>`` for agents created by AgentDesigner.
    """
    # Check core agents first
    cls = _CORE_AGENTS.get(agent_name.lower())
    if cls:
        return cls

    # Dynamic import for authored agents
    module_name = f"pipeline.agents.{agent_name}"
    try:
        mod = importlib.import_module(module_name)
        # Find the first BaseAgentWrapper subclass in the module
        from pipeline.agents.base import BaseAgentWrapper
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseAgentWrapper)
                and attr is not BaseAgentWrapper
            ):
                logger.info("Dynamically resolved agent: %s -> %s", agent_name, attr)
                return attr
    except ImportError:
        logger.debug("Could not import agent module: %s", module_name)
    except Exception as exc:
        logger.warning("Error resolving agent %s: %s", agent_name, exc)

    return None


class PipelineOrchestrator:
    """Orchestrates the pipeline using a declarative stage definition and
    adaptive meta-strategy.

    Key features:
        - **Data-driven stages**: pipeline composition loaded from config,
          modifiable at runtime by the PipelineArchitect agent.
        - **Adaptive meta-strategy**: retry counts, temperature escalation,
          and stall detection tuned from run history.
        - **Retry with escalation**: failed steps retry with increasing
          temperature (0.1 → 0.2 → 0.3 …).
        - **Stall detection**: if a Debugger produces the same experiment config
          as the previous attempt, inject a CRITICAL WARNING.
        - **Run history**: every stage attempt is logged for meta-strategy analysis.
    """

    def __init__(
        self,
        max_retries: int = 5,
        base_temperature: float = 0.1,
        temperature_step: float = 0.1,
        publish: bool = False,
    ) -> None:
        # Load adaptive meta-strategy; CLI args override defaults
        self.strategy = MetaStrategy.load()

        # CLI-specified values override the persisted strategy
        if max_retries != 5:
            self.strategy.max_retries = max_retries
        if base_temperature != 0.1:
            self.strategy.base_temperature = base_temperature
        if temperature_step != 0.1:
            self.strategy.temperature_step = temperature_step

        self.publish = publish
        self.run_id = str(uuid.uuid4())[:8]

    # Backwards-compatible property accessors
    @property
    def max_retries(self) -> int:
        return self.strategy.max_retries

    @property
    def base_temperature(self) -> float:
        return self.strategy.base_temperature

    @property
    def temperature_step(self) -> float:
        return self.strategy.temperature_step

    # ---------------------------------------------------------------- main
    def run(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute the pipeline using the current stage definition.

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

        pipeline = load_pipeline()
        # Filter publisher stage based on self.publish flag
        if not self.publish:
            pipeline = [s for s in pipeline if s.position != "publish"]

        total_stages = len(pipeline)
        _mon.reset()
        _mon.emit("pipeline", "pipeline_start", stages=total_stages)

        best: dict[str, Any] | None = None

        for stage_num, stage in enumerate(pipeline, 1):
            stage_label = f"STAGE {stage_num}: {stage.name.upper()}"
            logger.info("=== %s ===", stage_label)
            _mon.emit(
                "pipeline", "stage_start",
                stage=stage.name, stage_num=stage_num,
            )

            # Dispatch based on stage type
            if stage.name == "check_debug":
                # Special-case: the check/debug loop has its own internal logic
                check_debug_result = self._check_debug_loop(task)
                results["stages"].append(check_debug_result)
                results["success"] = check_debug_result.get("passed", False)
                continue

            # Resolve agent class
            agent_cls = resolve_agent(stage.agent_name)
            if agent_cls is None:
                if stage.optional:
                    logger.info(
                        "Skipping optional stage %s (agent %s not found)",
                        stage.name, stage.agent_name,
                    )
                    continue
                logger.error(
                    "Agent not found for required stage %s: %s",
                    stage.name, stage.agent_name,
                )
                results["stages"].append({
                    "agent": stage.agent_name,
                    "success": False,
                    "error": f"Agent not found: {stage.agent_name}",
                })
                _mon.emit("pipeline", "pipeline_complete", success=False)
                return results

            # Always set the stage's user_message (prevents bleed from prior stages)
            task["user_message"] = stage.user_message

            # Run the stage (with or without retry)
            if stage.retry:
                result = self._run_with_retry(agent_cls, task, stage.name)
            else:
                result = self._run_stage(agent_cls, task, stage)

            results["stages"].append({
                "agent": stage.agent_name,
                "success": result.success,
            })

            if not result.success:
                if stage.optional:
                    logger.warning(
                        "Optional stage %s failed, continuing", stage.name,
                    )
                    continue
                logger.error(
                    "%s failed — aborting pipeline. Last failure: %s",
                    stage.name,
                    result.raw_text[:5000],
                )
                # Persist the failed run so future planners can see what was tried
                self._save_to_hippius(results)
                _mon.emit("pipeline", "pipeline_complete", success=False)
                return results

            # Post-stage context extraction
            self._extract_stage_context(stage, result, task)

            # Track best experiment if extracted
            if stage.name == "trainer":
                best = self._extract_best(result)
                if best:
                    task["best_experiment"] = best.get("experiment", "")
                    task["best_metrics"] = best.get("metrics", "")
                    task["comparison"] = best.get("comparison", "")
                    task["experiment"] = best.get("experiment", "")

            if stage.position == "publish" and result.success:
                results["published"] = True
                if result.structured:
                    results["publish_info"] = result.structured

        # If we reached here without early return, the pipeline succeeded.
        # (check_debug sets success based on its own pass/fail; if there's
        # no check_debug stage in the pipeline we still mark success.)
        if "success" not in results or not results["success"]:
            # Only override if check_debug didn't already set it to True
            # Check if all required stages passed
            all_passed = all(
                s.get("success", False) or s.get("passed", False)
                for s in results["stages"]
            )
            if all_passed:
                results["success"] = True

        # Attach best experiment info to the summary
        if best:
            results["best_experiment"] = best.get("experiment")
            results["best_crps"] = best.get("crps")
            if best.get("metrics"):
                results["best_metrics"] = best["metrics"]
            if best.get("comparison"):
                results["comparison"] = best["comparison"]

        # Persist pipeline summary to Hippius
        self._save_to_hippius(results)

        _mon.emit(
            "pipeline", "pipeline_complete",
            success=results.get("success", False),
            best_crps=results.get("best_crps"),
        )

        return results

    # ----------------------------------------------------------- single stage
    def _run_stage(
        self,
        agent_cls: type,
        task: dict[str, Any],
        stage: StageSpec,
    ) -> AgentResult:
        """Run a single stage without retry logic."""
        params = self.strategy.for_stage(stage.name)
        temp = params["base_temperature"]
        agent = agent_cls(temperature=temp)

        try:
            result = agent.run(task)
        except Exception as exc:
            logger.exception("%s crashed with unhandled exception", stage.name)
            result = AgentResult(
                success=False,
                raw_text=f"Agent crashed: {type(exc).__name__}: {exc}",
            )

        record_run_event(
            pipeline_run_id=self.run_id,
            stage=stage.name,
            attempt=1,
            temperature=temp,
            success=result.success,
            error=result.raw_text[:3000] if not result.success else None,
        )

        return result

    # ----------------------------------------------------------- retry wrapper
    @staticmethod
    def _is_non_recoverable(result: AgentResult) -> bool:
        """Check if an agent result indicates a non-recoverable error.

        Detects environment errors (missing dependencies, broken imports) that
        cannot be fixed by retrying with higher temperature.  Tools signal this
        via ``"error_type": "environment"`` and ``"recoverable": false`` in their
        structured output.

        Infrastructure errors (``"error_type": "infrastructure"``) are explicitly
        **recoverable** — the agent should retry with a fresh deployment.
        """
        # Check structured result dict
        if isinstance(result.structured, dict):
            nested = result.structured.get("result", result.structured)
            if isinstance(nested, dict):
                # Explicitly recoverable errors (infrastructure, etc.)
                if nested.get("recoverable") is True:
                    return False
                if nested.get("error_type") == "environment":
                    return True

        # Check raw text for the JSON markers (agent may embed them in summary)
        raw = result.raw_text or ""
        if '"error_type": "environment"' in raw and '"recoverable": false' in raw:
            return True

        return False

    def _run_with_retry(
        self,
        agent_cls: type,
        task: dict[str, Any],
        stage_name: str,
    ) -> AgentResult:
        """Run an agent with escalating temperature on failure."""
        params = self.strategy.for_stage(stage_name)
        max_retries = params["max_retries"]
        base_temp = params["base_temperature"]
        temp_step = params["temperature_step"]
        last_result: AgentResult | None = None
        prior_attempts: list[str] = []

        for attempt in range(max_retries):
            temp = base_temp + (attempt * temp_step)
            logger.info(
                "%s attempt %d/%d (temp=%.2f)",
                stage_name,
                attempt + 1,
                max_retries,
                temp,
            )
            _mon.emit(
                "pipeline", "retry_attempt",
                stage=stage_name, attempt=attempt + 1,
                max_attempts=max_retries, temperature=temp,
            )

            agent = agent_cls(temperature=temp)

            # Inject failure context from previous attempt, or use default
            if last_result and not last_result.success:
                # Build a summary of ALL previous attempts so the agent
                # knows exactly what was tried and avoids repeating it.
                attempts_summary = "\n".join(
                    f"  Attempt {i}: {desc}" for i, desc in enumerate(prior_attempts, 1)
                )
                task["user_message"] = (
                    f"## Previous Attempts (all failed)\n\n"
                    f"{attempts_summary}\n\n"
                    f"## Most Recent Error\n\n{last_result.raw_text[:3000]}\n\n"
                    f"You MUST try a DIFFERENT approach. Do NOT repeat the same "
                    f"experiments or GPU models listed above. "
                    f"Attempt {attempt + 1}/{max_retries}."
                )

            try:
                result = agent.run(task)
            except Exception as exc:
                logger.exception(
                    "%s attempt %d crashed with unhandled exception",
                    stage_name,
                    attempt + 1,
                )
                result = AgentResult(
                    success=False,
                    raw_text=f"Agent crashed: {type(exc).__name__}: {exc}",
                )

            record_run_event(
                pipeline_run_id=self.run_id,
                stage=stage_name,
                attempt=attempt + 1,
                temperature=temp,
                success=result.success,
                error=result.raw_text[:3000] if not result.success else None,
            )

            if result.success:
                return result

            # Extract a compact summary of what this attempt tried
            prior_attempts.append(self._summarize_attempt(result))

            last_result = result
            logger.warning(
                "%s attempt %d failed: %s",
                stage_name,
                attempt + 1,
                result.raw_text[:3000],
            )

            # Stop retrying if the error is non-recoverable (e.g. missing dependency)
            if self._is_non_recoverable(result):
                logger.error(
                    "%s: non-recoverable error detected — skipping remaining %d retries",
                    stage_name,
                    self.max_retries - attempt - 1,
                )
                _mon.emit(
                    "pipeline", "non_recoverable_error",
                    stage=stage_name, attempt=attempt + 1,
                )
                break

        return last_result or AgentResult(success=False, raw_text="No attempts made")

    # ----------------------------------------------------------- check/debug
    def _check_debug_loop(self, task: dict[str, Any]) -> dict[str, Any]:
        """Alternating CodeChecker → Debugger loop with stall detection."""
        params = self.strategy.for_stage("check_debug")
        max_retries = params["max_retries"]
        base_temp = params["base_temperature"]
        temp_step = params["temperature_step"]
        stall_threshold = params["stall_threshold"]

        prev_experiment_json: str | None = None
        stall_count = 0
        stage_results: list[dict] = []

        for attempt in range(max_retries):
            # Check
            checker = CodeCheckerAgent(temperature=base_temp)
            task["user_message"] = "Validate the best experiment config and its results."
            try:
                check_result = checker.run(task)
            except Exception as exc:
                logger.exception("CodeChecker crashed on attempt %d", attempt + 1)
                check_result = AgentResult(
                    success=False,
                    raw_text=f"CodeChecker crashed: {type(exc).__name__}: {exc}",
                )

            stage_results.append({
                "agent": "codechecker",
                "attempt": attempt,
                "success": check_result.success,
            })

            record_run_event(
                pipeline_run_id=self.run_id,
                stage="codechecker",
                attempt=attempt + 1,
                temperature=base_temp,
                success=check_result.success,
            )

            if check_result.success:
                structured = check_result.structured or {}
                passed = structured.get("success", True) if isinstance(structured, dict) else True
                if passed:
                    return {"passed": True, "attempts": attempt + 1, "stages": stage_results}

            # Debug
            temp = base_temp + (attempt * temp_step)
            logger.info("CodeChecker failed — running Debugger (attempt %d)", attempt + 1)
            task["error_report"] = check_result.raw_text[:10000]

            # Stall detection: compare experiment JSON across attempts
            current_exp = task.get("best_experiment", "")
            if current_exp and current_exp == prev_experiment_json:
                stall_count += 1
                if stall_count >= stall_threshold:
                    logger.warning("STALL DETECTED: experiment config unchanged")
                    _mon.emit("pipeline", "stall_detected", attempt=attempt + 1)
                    task["user_message"] = (
                        "CRITICAL WARNING: The experiment config has NOT changed since the last "
                        "attempt. You MUST take a DIFFERENT approach — change blocks, head, "
                        "d_model, or learning rate.\n\n"
                        f"Error report:\n{task['error_report']}"
                    )
                else:
                    task["user_message"] = (
                        f"Fix the experiment. Error report:\n{task['error_report']}"
                    )
            else:
                stall_count = 0
                task["user_message"] = f"Fix the experiment. Error report:\n{task['error_report']}"
            prev_experiment_json = current_exp

            debugger = DebuggerAgent(temperature=temp)
            try:
                debug_result = debugger.run(task)
            except Exception as exc:
                logger.exception("Debugger crashed on attempt %d", attempt + 1)
                debug_result = AgentResult(
                    success=False,
                    raw_text=f"Debugger crashed: {type(exc).__name__}: {exc}",
                )
            stage_results.append({
                "agent": "debugger",
                "attempt": attempt,
                "success": debug_result.success,
            })

            record_run_event(
                pipeline_run_id=self.run_id,
                stage="debugger",
                attempt=attempt + 1,
                temperature=temp,
                success=debug_result.success,
            )

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

        return {"passed": False, "attempts": max_retries, "stages": stage_results}

    # ----------------------------------------------------------- context extraction
    @staticmethod
    def _extract_stage_context(
        stage: StageSpec,
        result: AgentResult,
        task: dict[str, Any],
    ) -> None:
        """Extract useful context from a stage result and inject into the task.

        The planner's output is extracted into task["plan"] for downstream agents.
        Custom stages' structured output is stored under their name.
        """
        if stage.name == "planner" and result.success:
            plan_data = result.structured or {}
            task["plan"] = (
                json.dumps(plan_data, indent=2)
                if isinstance(plan_data, dict)
                else str(plan_data)
            )
            logger.info("Plan passed to trainer: %s", task["plan"][:500])
        elif result.success and result.structured:
            # Store custom stage output for downstream consumption
            task[f"{stage.name}_result"] = result.structured

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

            # Persist the full comparison ranking if available, otherwise
            # fall back to a minimal snapshot with just the best experiment.
            comparison = results.get("comparison")
            if comparison:
                # Full trainer ranking (JSON string or dict)
                comp_data = (
                    json.loads(comparison)
                    if isinstance(comparison, str)
                    else comparison
                )
                comp_data.setdefault(
                    "best_crps", results.get("best_crps")
                )
                comp_data.setdefault("success", results.get("success", False))
                save_comparison(comp_data)
            elif results.get("best_experiment"):
                save_comparison({
                    "best_experiment": results["best_experiment"],
                    "best_crps": results.get("best_crps"),
                    "success": results.get("success", False),
                })

            # Reset run ID so next pipeline invocation gets a fresh one
            reset_run_id()
        except Exception as exc:
            logger.debug("Hippius save skipped: %s", exc)

    # ----------------------------------------------------------- helpers
    @staticmethod
    def _summarize_attempt(result: AgentResult) -> str:
        """Extract a compact description of what an attempt tried (for retry context)."""
        parts: list[str] = []

        # Try to pull experiment names / GPU models from structured output
        if isinstance(result.structured, dict):
            nested = result.structured.get("result", result.structured)
            if isinstance(nested, dict):
                # Experiments tried
                exps = nested.get("validated_experiments") or nested.get("experiments")
                if isinstance(exps, list):
                    names = [
                        e.get("name", "?") for e in exps
                        if isinstance(e, dict)
                    ]
                    if names:
                        parts.append(f"experiments=[{', '.join(names[:6])}]")

                # GPU models tried
                gpus = nested.get("gpu_models_tried")
                if isinstance(gpus, list) and gpus:
                    parts.append(f"gpu_models=[{', '.join(str(g) for g in gpus[:6])}]")

                # Blocker type
                blocker = nested.get("infrastructure_blocker")
                if isinstance(blocker, dict):
                    parts.append(f"error={blocker.get('error', 'unknown')}")
                elif nested.get("error"):
                    parts.append(f"error={nested['error']}")

        if not parts:
            # Fallback: first line of raw text
            first_line = (result.raw_text or "unknown").split("\n")[0][:200]
            parts.append(first_line)

        return "; ".join(parts)

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
