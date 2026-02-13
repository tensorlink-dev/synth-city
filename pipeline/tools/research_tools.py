"""
Research tools — wrap the open-synth-miner ResearchSession API as agent tools.

These tools give agents access to the full experiment lifecycle:
  discover components → create experiment → validate → run → compare
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
from pipeline.tools.registry import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singleton ResearchSession
# ---------------------------------------------------------------------------
_session = None


def _get_session():
    global _session
    if _session is None:
        from src.research.agent_api import ResearchSession
        _session = ResearchSession()
    return _session


# ---------------------------------------------------------------------------
# Discovery tools
# ---------------------------------------------------------------------------

@tool(description="List all available backbone blocks with their parameters, strengths, and compute cost.")
def list_blocks() -> str:
    """Discover registered blocks in open-synth-miner."""
    session = _get_session()
    blocks = session.list_blocks()
    return json.dumps(blocks, indent=2)


@tool(description="List all available head types with their parameters and expressiveness levels.")
def list_heads() -> str:
    """Discover registered heads in open-synth-miner."""
    session = _get_session()
    heads = session.list_heads()
    return json.dumps(heads, indent=2)


@tool(description="List all ready-to-run presets with their block+head combinations and tags.")
def list_presets() -> str:
    """Discover built-in experiment presets."""
    session = _get_session()
    presets = session.list_presets()
    return json.dumps(presets, indent=2)


# ---------------------------------------------------------------------------
# Experiment construction
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Create an experiment config from blocks and a head. "
        "blocks: JSON list of block names (e.g. [\"TransformerBlock\", \"LSTMBlock\"]). "
        "head: head name (e.g. \"GBMHead\"). "
        "Optional overrides: d_model, horizon, n_paths, lr, seq_len, batch_size, feature_dim as JSON."
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
            "block_kwargs": {"type": "string", "description": "Per-block extra params as JSON list of dicts"},
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
    session = _get_session()
    try:
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
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


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
    session = _get_session()
    try:
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
    session = _get_session()
    try:
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
    session = _get_session()
    try:
        exp = json.loads(experiment) if isinstance(experiment, str) else experiment
        result = session.run(exp, epochs=epochs, name=name or None)
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


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
    session = _get_session()
    try:
        ov = json.loads(overrides) if overrides else None
        result = session.run_preset(preset_name, epochs=epochs, overrides=ov)
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


@tool(
    description=(
        "Sweep multiple presets and return comparison. "
        "preset_names: JSON array of names, or empty for all presets."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "preset_names": {"type": "string", "description": "JSON array of preset names (empty = all)"},
            "epochs": {"type": "integer", "description": "Epochs per preset"},
        },
        "required": [],
    },
)
def sweep_presets(preset_names: str = "", epochs: int = RESEARCH_EPOCHS) -> str:
    """Sweep presets and return ranked comparison."""
    session = _get_session()
    try:
        names = json.loads(preset_names) if preset_names else None
        result = session.sweep(preset_names=names, epochs=epochs)
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@tool(description="Compare all experiment results accumulated in the current session. Returns ranking sorted by CRPS (best first).")
def compare_results() -> str:
    """Compare all results from this session."""
    session = _get_session()
    try:
        result = session.compare()
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(description="Get a summary of all experiments run in the current session.")
def session_summary() -> str:
    """Get session summary: num experiments, comparison, all results."""
    session = _get_session()
    try:
        result = session.summary()
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(description="Clear the research session (reset accumulated results). Use between unrelated experiment batches.")
def clear_session() -> str:
    """Reset the research session."""
    session = _get_session()
    session.clear()
    return json.dumps({"status": "session cleared"})
