"""
Research tools — wrap the open-synth-miner ResearchSession API as agent tools.

These tools give agents access to the full experiment lifecycle:
  discover components → create experiment → validate → run → compare

Memory management:
  The ResearchSession accumulates results in-memory.  To prevent unbounded
  growth during long-running sessions, ``flush_session`` offloads the current
  results to Hippius storage and clears all but the top-N experiments (by CRPS)
  from memory.  The Trainer can call this periodically, and the orchestrator
  triggers it automatically when the result count exceeds SESSION_MAX_RESULTS.
"""

from __future__ import annotations

import json
import logging
import traceback

from config import (
    RESEARCH_BATCH_SIZE,
    RESEARCH_D_MODEL,
    RESEARCH_EPOCHS,
    RESEARCH_FEATURE_DIM,
    RESEARCH_HORIZON,
    RESEARCH_LR,
    RESEARCH_N_PATHS,
    RESEARCH_SEQ_LEN,
)
from pipeline.monitor import get_monitor
from pipeline.tools.registry import tool

logger = logging.getLogger(__name__)
_mon = get_monitor()

# Maximum results to keep in memory before auto-flushing
SESSION_MAX_RESULTS: int = 100
# How many top results (by CRPS) to retain after a flush
SESSION_KEEP_TOP_N: int = 10

# ---------------------------------------------------------------------------
# Lazy-loaded ResearchSession — per-bot when running via bridge, global for CLI
# ---------------------------------------------------------------------------
_session = None


def _get_session():
    """Return the ResearchSession for the current bot, or the global fallback."""
    from integrations.openclaw.bot_sessions import get_current_session

    bot = get_current_session()
    if bot is not None:
        return bot.get_research_session()

    # CLI / standalone fallback
    global _session
    if _session is None:
        from src.research.agent_api import ResearchSession
        _session = ResearchSession()
    return _session


def _maybe_auto_flush():
    """Flush the session to Hippius if it's grown past the threshold."""
    try:
        session = _get_session()
        summary = session.summary()
        count = summary.get("num_experiments", 0) if isinstance(summary, dict) else 0
        if count >= SESSION_MAX_RESULTS:
            logger.info(
                "Session has %d results (threshold %d) — auto-flushing to Hippius",
                count, SESSION_MAX_RESULTS,
            )
            _mon.emit("system", "auto_flush", count=count, threshold=SESSION_MAX_RESULTS)
            _do_flush(keep_top_n=SESSION_KEEP_TOP_N)
    except Exception as exc:
        logger.debug("Auto-flush check failed: %s", exc)


def _do_flush(keep_top_n: int = SESSION_KEEP_TOP_N) -> dict:
    """Save all session results to Hippius, then clear and keep only top-N."""
    session = _get_session()
    comparison = {}
    saved_count = 0

    try:
        comparison = session.compare()
    except Exception:
        pass

    # Save comparison snapshot to Hippius
    try:
        from pipeline.tools.hippius_store import save_comparison
        if comparison:
            save_comparison(comparison)
    except Exception:
        pass

    # Get the ranking so we know what to keep
    ranking = comparison.get("ranking", []) if isinstance(comparison, dict) else []
    saved_count = len(ranking)

    # Clear the in-memory session
    session.clear()
    logger.info("Flushed %d results from session, cleared memory", saved_count)

    # Re-create the top-N experiments so agents still have a baseline to compare against
    # (These are lightweight config dicts, not full model weights)
    restored = 0
    for entry in ranking[:keep_top_n]:
        try:
            exp_config = entry.get("experiment") or entry.get("config")
            if exp_config and isinstance(exp_config, dict):
                session.create_experiment(
                    blocks=exp_config.get("model", {}).get("backbone", {}).get("blocks", []),
                    head=(
                        exp_config.get("model", {})
                        .get("head", {})
                        .get("_target_", "GBMHead")
                        .split(".")[-1]
                    ),
                    d_model=exp_config.get("model", {}).get("backbone", {}).get("d_model", 32),
                )
                restored += 1
        except Exception:
            pass

    return {
        "flushed": saved_count,
        "kept_in_memory": restored,
        "hippius_saved": bool(comparison),
    }


# ---------------------------------------------------------------------------
# Discovery tools
# ---------------------------------------------------------------------------

@tool(
    description=(
        "List all available backbone blocks with their"
        " parameters, strengths, and compute cost."
    ),
)
def list_blocks() -> str:
    """Discover registered blocks in open-synth-miner."""
    try:
        session = _get_session()
        blocks = session.list_blocks()
        return json.dumps(blocks, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(description="List all available head types with their parameters and expressiveness levels.")
def list_heads() -> str:
    """Discover registered heads in open-synth-miner."""
    try:
        session = _get_session()
        heads = session.list_heads()
        return json.dumps(heads, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(description="List all ready-to-run presets with their block+head combinations and tags.")
def list_presets() -> str:
    """Discover built-in experiment presets."""
    try:
        session = _get_session()
        presets = session.list_presets()
        return json.dumps(presets, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------------
# Experiment construction
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Create an experiment config from blocks and a head. "
        "blocks: JSON list of block names (e.g. [\"TransformerBlock\", \"LSTMBlock\"]). "
        "head: head name (e.g. \"GBMHead\"). "
        "Optional overrides: d_model, horizon, n_paths, lr, "
        "seq_len, batch_size, feature_dim as JSON."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "blocks": {"type": "string", "description": "JSON array of block names"},
            "head": {"type": "string", "description": "Head name (default: GBMHead)"},
            "d_model": {"type": "integer", "description": "Hidden dimension (default: 32)"},
            "horizon": {"type": "integer", "description": "Prediction steps (default: 12)"},
            "n_paths": {"type": "integer", "description": "Monte Carlo paths (default: 100)"},
            "lr": {"type": "number", "description": "Learning rate (default: 0.001)"},
            "seq_len": {"type": "integer", "description": "Input sequence length (default: 32)"},
            "batch_size": {"type": "integer", "description": "Batch size (default: 4)"},
            "feature_dim": {"type": "integer", "description": "Input features (default: 4)"},
            "head_kwargs": {"type": "string", "description": "Extra head params as JSON dict"},
            "block_kwargs": {
                "type": "string",
                "description": "Per-block extra params as JSON list of dicts",
            },
        },
        "required": ["blocks"],
    },
)
def create_experiment(
    blocks: str,
    head: str = "GBMHead",
    d_model: int = RESEARCH_D_MODEL,
    horizon: int = RESEARCH_HORIZON,
    n_paths: int = RESEARCH_N_PATHS,
    lr: float = RESEARCH_LR,
    seq_len: int = RESEARCH_SEQ_LEN,
    batch_size: int = RESEARCH_BATCH_SIZE,
    feature_dim: int = RESEARCH_FEATURE_DIM,
    head_kwargs: str = "",
    block_kwargs: str = "",
) -> str:
    """Create an experiment configuration dict."""
    try:
        session = _get_session()
        block_list = json.loads(blocks) if isinstance(blocks, str) else blocks
        hk = json.loads(head_kwargs) if head_kwargs else None
        bk = json.loads(block_kwargs) if block_kwargs else None

        experiment = session.create_experiment(
            blocks=block_list,
            head=head,
            d_model=d_model,
            feature_dim=feature_dim,
            seq_len=seq_len,
            horizon=horizon,
            n_paths=n_paths,
            batch_size=batch_size,
            lr=lr,
            head_kwargs=hk,
            block_kwargs=bk,
        )
        return json.dumps(experiment, indent=2)
    except Exception as exc:
        return json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Validate an experiment config without executing it. "
        "Returns param count, errors, and warnings. "
        "experiment: the experiment config as a JSON string."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "experiment": {"type": "string", "description": "Experiment config JSON"},
        },
        "required": ["experiment"],
    },
)
def validate_experiment(experiment: str) -> str:
    """Validate an experiment config (no execution)."""
    try:
        session = _get_session()
        exp = json.loads(experiment) if isinstance(experiment, str) else experiment
        result = session.validate(exp)
        return json.dumps(result, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Describe an experiment: blocks, head, param count, training config, validation. "
        "experiment: the experiment config as a JSON string."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "experiment": {"type": "string", "description": "Experiment config JSON"},
        },
        "required": ["experiment"],
    },
)
def describe_experiment(experiment: str) -> str:
    """Get a full description of an experiment."""
    try:
        session = _get_session()
        exp = json.loads(experiment) if isinstance(experiment, str) else experiment
        result = session.describe(exp)
        return json.dumps(result, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Run an experiment: train the model and return metrics including CRPS. "
        "experiment: the experiment config as a JSON string. "
        "epochs: number of training epochs (default from config). "
        "Experiments never raise — errors come back in the result dict."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "experiment": {"type": "string", "description": "Experiment config JSON"},
            "epochs": {"type": "integer", "description": "Training epochs (default: config)"},
            "name": {"type": "string", "description": "Optional experiment name"},
        },
        "required": ["experiment"],
    },
)
def run_experiment(experiment: str, epochs: int = RESEARCH_EPOCHS, name: str = "") -> str:
    """Run an experiment and return metrics."""
    try:
        session = _get_session()
        exp = json.loads(experiment) if isinstance(experiment, str) else experiment
        _mon.emit("experiment", "experiment_start", name=name)
        result = session.run(exp, epochs=epochs, name=name or None)

        # Auto-save to Hippius if configured
        try:
            from pipeline.tools.hippius_store import save_experiment_result
            label = name or f"exp-{len(session.compare().get('ranking', []))}"
            save_experiment_result(label, exp, result)
        except Exception:
            pass  # Storage is best-effort, never block the pipeline

        # Emit experiment result for dashboard monitoring
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        _mon.emit(
            "experiment", "experiment_result",
            name=name,
            crps=metrics.get("crps"),
            status=result.get("status", "unknown") if isinstance(result, dict) else "unknown",
        )

        # Auto-flush if session has grown too large
        _maybe_auto_flush()

        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        _mon.emit("system", "error", message=f"run_experiment failed: {exc}")
        return json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


@tool(
    description=(
        "Run a built-in preset by name. Optionally override training parameters. "
        "preset_name: e.g. 'transformer_lstm', 'pure_transformer', 'conv_gru'. "
        "overrides: JSON dict like {\"training\": {\"horizon\": 24, \"lr\": 0.0001}}."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "preset_name": {"type": "string", "description": "Preset name"},
            "epochs": {"type": "integer", "description": "Training epochs"},
            "overrides": {"type": "string", "description": "Overrides as JSON dict"},
        },
        "required": ["preset_name"],
    },
)
def run_preset(preset_name: str, epochs: int = RESEARCH_EPOCHS, overrides: str = "") -> str:
    """Run a preset experiment."""
    try:
        session = _get_session()
        ov = json.loads(overrides) if overrides else None
        result = session.run_preset(preset_name, epochs=epochs, overrides=ov)
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        return json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


@tool(
    description=(
        "Sweep multiple presets and return comparison. "
        "preset_names: JSON array of names, or empty for all presets."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "preset_names": {
                "type": "string",
                "description": "JSON array of preset names (empty = all)",
            },
            "epochs": {"type": "integer", "description": "Epochs per preset"},
        },
        "required": [],
    },
)
def sweep_presets(preset_names: str = "", epochs: int = RESEARCH_EPOCHS) -> str:
    """Sweep presets and return ranked comparison."""
    try:
        session = _get_session()
        names = json.loads(preset_names) if preset_names else None
        result = session.sweep(preset_names=names, epochs=epochs)
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        return json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Compare all experiment results accumulated in the"
        " current session. Returns ranking sorted by CRPS"
        " (best first)."
    ),
)
def compare_results() -> str:
    """Compare all results from this session."""
    try:
        session = _get_session()
        result = session.compare()
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(description="Get a summary of all experiments run in the current session.")
def session_summary() -> str:
    """Get session summary: num experiments, comparison, all results."""
    try:
        session = _get_session()
        result = session.summary()
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Clear the research session (reset accumulated results)."
        " Use between unrelated experiment batches."
    ),
)
def clear_session() -> str:
    """Reset the research session."""
    try:
        session = _get_session()
        session.clear()
        return json.dumps({"status": "session cleared"})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Flush session results to Hippius storage and free memory. "
        "Saves all current results to persistent storage, clears the in-memory session, "
        "and keeps only the top-N experiments (by CRPS) in memory for comparison. "
        "Use this periodically during long training runs to prevent out-of-memory. "
        "All flushed results remain accessible via load_hippius_history. "
        "keep_top_n: how many best results to retain in memory (default 10)."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "keep_top_n": {
                "type": "integer",
                "description": "Keep top N results in memory (default 10)",
            },
        },
        "required": [],
    },
)
def flush_session(keep_top_n: int = SESSION_KEEP_TOP_N) -> str:
    """Flush session to Hippius and free memory, keeping top-N."""
    try:
        result = _do_flush(keep_top_n=keep_top_n)
        return json.dumps({"status": "flushed", **result}, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
