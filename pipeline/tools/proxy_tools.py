"""Low-cost proxy tools for architecture reasoning and hyperparameter screening.

These tools give agents cheap signals to evaluate architecture choices BEFORE
committing to full GPU training runs.  They fall into two categories:

**Local (zero-cost, no GPU needed)**
- ``estimate_params`` — parameter count from config (formula-based)
- ``estimate_flops`` — relative compute cost estimate
- ``generate_ablation_configs`` — systematic variant generation
- ``sweep_configs`` — grid/random search over hyperparameter space

**Remote (low-cost, run on existing deployment)**
- ``probe_architecture`` — forward pass + single-batch gradient check on GPU
  Uses the ``POST /probe`` endpoint on the training server.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import random
from typing import Any

from config import (
    RESEARCH_BATCH_SIZE,
    RESEARCH_D_MODEL,
    RESEARCH_FEATURE_DIM,
    RESEARCH_HORIZON,
    RESEARCH_LR,
    RESEARCH_N_PATHS,
    RESEARCH_SEQ_LEN,
    TIMEFRAME_CONFIGS,
)
from pipeline.tools.registry import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Block metadata for cost estimation
# ---------------------------------------------------------------------------

# Approximate parameter multipliers relative to d_model^2.
# These are rough estimates based on typical layer implementations.
_BLOCK_PARAM_FACTORS: dict[str, float] = {
    "RevIN": 2.0,              # 2 * d_model (affine scale + bias)
    "LayerNormBlock": 2.0,     # 2 * d_model (affine scale + bias)
    "DLinearBlock": 1.0,       # ~d_model * d_model (linear decomposition)
    "RNNBlock": 4.0,           # ~4 * d_model^2 (input, hidden, bias)
    "ResConvBlock": 3.0,       # conv kernels + residual projection
    "BiTCNBlock": 6.0,         # multi-layer dilated convolutions
    "SDEEvolutionBlock": 4.0,  # drift + diffusion networks
    "GRUBlock": 6.0,           # 3 gates * 2 * d_model^2
    "LSTMBlock": 8.0,          # 4 gates * 2 * d_model^2
    "FourierBlock": 4.0,       # FFT + learned filter + inverse FFT
    "TransformerBlock": 8.0,   # Q,K,V projections + FFN (4x expansion)
    "TimeMixerBlock": 6.0,     # multi-scale mixing layers
    "Unet1DBlock": 10.0,       # encoder + decoder + skip connections
    "TransformerEncoder": 12.0,  # deeper multi-head attention + FFN
    "TimesNetBlock": 14.0,     # 2D convolution + period detection
}

# FLOPs multiplier per token (relative to param count).
# Attention-based blocks scale with seq_len; others are constant.
_BLOCK_FLOP_SCALING: dict[str, str] = {
    "RevIN": "constant",
    "LayerNormBlock": "constant",
    "DLinearBlock": "constant",
    "RNNBlock": "sequential",      # O(seq_len * d_model^2)
    "ResConvBlock": "constant",
    "BiTCNBlock": "constant",      # dilated = O(log(seq_len)) layers
    "SDEEvolutionBlock": "constant",
    "GRUBlock": "sequential",
    "LSTMBlock": "sequential",
    "FourierBlock": "nlogn",       # FFT = O(seq_len * log(seq_len))
    "TransformerBlock": "quadratic",  # O(seq_len^2 * d_model)
    "TimeMixerBlock": "constant",
    "Unet1DBlock": "constant",
    "TransformerEncoder": "quadratic",
    "TimesNetBlock": "quadratic",
}

_HEAD_PARAM_FACTORS: dict[str, float] = {
    "GBMHead": 2.0,            # linear mu + sigma
    "SDEHead": 6.0,            # deeper mu/sigma networks
    "SimpleHorizonHead": 4.0,  # per-step pooling + projection
    "HorizonHead": 10.0,       # cross-attention per step
    "NeuralBridgeHead": 12.0,  # macro + micro networks
    "NeuralSDEHead": 16.0,     # full neural SDE solver
}


def _estimate_block_params(block_name: str, d_model: int) -> int:
    """Estimate parameter count for a single block."""
    factor = _BLOCK_PARAM_FACTORS.get(block_name, 6.0)
    if block_name in ("RevIN", "LayerNormBlock"):
        return int(factor * d_model)
    return int(factor * d_model * d_model)


def _estimate_head_params(head_name: str, d_model: int, horizon: int) -> int:
    """Estimate parameter count for a head."""
    factor = _HEAD_PARAM_FACTORS.get(head_name, 6.0)
    base = int(factor * d_model * d_model)
    # Horizon-aware heads scale with prediction length
    if head_name in ("HorizonHead", "NeuralBridgeHead", "NeuralSDEHead"):
        base += d_model * horizon
    return base


def _relative_flops(
    blocks: list[str], head: str, d_model: int, seq_len: int, horizon: int,
) -> float:
    """Estimate relative FLOPs (arbitrary units, useful for comparison only)."""
    total = 0.0
    # Input projection: feature_dim -> d_model, applied per timestep
    input_proj_params = RESEARCH_FEATURE_DIM * d_model + d_model
    total += input_proj_params * seq_len
    for block in blocks:
        params = _estimate_block_params(block, d_model)
        scaling = _BLOCK_FLOP_SCALING.get(block, "constant")
        if scaling == "constant":
            total += params * seq_len
        elif scaling == "sequential":
            total += params * seq_len
        elif scaling == "nlogn":
            total += params * seq_len * max(1, math.log2(seq_len))
        elif scaling == "quadratic":
            total += params * seq_len + d_model * seq_len * seq_len
    # Head FLOPs
    head_params = _estimate_head_params(head, d_model, horizon)
    total += head_params * horizon
    return total


# ---------------------------------------------------------------------------
# Local proxy tools (zero-cost)
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Estimate parameter count and memory footprint for an architecture "
        "WITHOUT instantiating a model. Returns per-block and total param counts, "
        "estimated GPU memory, and relative cost tier. "
        "Use this to compare architectures before committing to GPU training. "
        "blocks: JSON list of block names. head: head name."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "blocks": {"type": "string", "description": "JSON array of block names"},
            "head": {"type": "string", "description": "Head name (default: GBMHead)"},
            "d_model": {"type": "integer", "description": "Hidden dimension (default: 32)"},
            "horizon": {"type": "integer", "description": "Prediction steps (default: 12)"},
            "n_paths": {"type": "integer", "description": "Monte Carlo paths (default: 100)"},
        },
        "required": ["blocks"],
    },
)
def estimate_params(
    blocks: str,
    head: str = "GBMHead",
    d_model: int = RESEARCH_D_MODEL,
    horizon: int = RESEARCH_HORIZON,
    n_paths: int = RESEARCH_N_PATHS,
) -> str:
    """Estimate parameter count and memory for an architecture config."""
    try:
        block_list = json.loads(blocks) if isinstance(blocks, str) else blocks

        breakdown: list[dict[str, Any]] = []
        total_params = 0

        # Input projection: feature_dim -> d_model
        input_proj = RESEARCH_FEATURE_DIM * d_model + d_model  # weight + bias
        breakdown.append({"layer": "input_projection", "params": input_proj})
        total_params += input_proj

        for block_name in block_list:
            params = _estimate_block_params(block_name, d_model)
            breakdown.append({"layer": block_name, "params": params})
            total_params += params

        head_params = _estimate_head_params(head, d_model, horizon)
        breakdown.append({"layer": head, "params": head_params})
        total_params += head_params

        # Memory estimate: params * 4 bytes (float32) * 3 (params + grads + optimizer)
        memory_mb = (total_params * 4 * 3) / (1024 * 1024)
        # Activation memory (rough): batch_size * seq_len * d_model * 4 * num_layers
        act_memory_mb = (
            RESEARCH_BATCH_SIZE * RESEARCH_SEQ_LEN * d_model * 4 * len(block_list)
        ) / (1024 * 1024)

        # Cost tier
        if total_params < 50_000:
            tier = "very low"
        elif total_params < 200_000:
            tier = "low"
        elif total_params < 1_000_000:
            tier = "medium"
        elif total_params < 5_000_000:
            tier = "high"
        else:
            tier = "very high"

        return json.dumps({
            "total_params": total_params,
            "total_params_human": (
                f"{total_params / 1_000_000:.2f}M" if total_params >= 1_000_000
                else f"{total_params / 1_000:.1f}K"
            ),
            "breakdown": breakdown,
            "estimated_gpu_memory_mb": round(memory_mb + act_memory_mb, 1),
            "parameter_memory_mb": round(memory_mb, 1),
            "activation_memory_mb": round(act_memory_mb, 1),
            "cost_tier": tier,
            "config": {
                "blocks": block_list,
                "head": head,
                "d_model": d_model,
                "horizon": horizon,
            },
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Compare relative compute cost (FLOPs) across multiple architectures. "
        "Takes a list of architecture specs and returns them ranked by estimated cost. "
        "Use this to pick the cheapest architectures for initial screening. "
        "architectures: JSON array of {blocks, head, d_model} objects."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "architectures": {
                "type": "string",
                "description": (
                    "JSON array of architecture specs, each with "
                    "'blocks' (list), 'head' (str), optional 'd_model' (int)"
                ),
            },
        },
        "required": ["architectures"],
    },
)
def estimate_flops(architectures: str) -> str:
    """Compare relative compute cost across architectures."""
    try:
        arch_list = json.loads(architectures) if isinstance(architectures, str) else architectures

        results = []
        for i, arch in enumerate(arch_list):
            blocks = arch.get("blocks", [])
            head = arch.get("head", "GBMHead")
            d_model = arch.get("d_model", RESEARCH_D_MODEL)
            seq_len = arch.get("seq_len", RESEARCH_SEQ_LEN)
            horizon = arch.get("horizon", RESEARCH_HORIZON)

            flops = _relative_flops(blocks, head, d_model, seq_len, horizon)
            params = sum(_estimate_block_params(b, d_model) for b in blocks)
            params += _estimate_head_params(head, d_model, horizon)

            results.append({
                "index": i,
                "blocks": blocks,
                "head": head,
                "d_model": d_model,
                "estimated_params": params,
                "relative_flops": round(flops),
            })

        # Sort by FLOPs (cheapest first)
        results.sort(key=lambda x: x["relative_flops"])

        # Normalise to cheapest = 1.0x
        if results and results[0]["relative_flops"] > 0:
            base = results[0]["relative_flops"]
            for r in results:
                r["cost_ratio"] = round(r["relative_flops"] / base, 2)

        return json.dumps({
            "ranked_by_cost": results,
            "cheapest": results[0] if results else None,
            "most_expensive": results[-1] if results else None,
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Generate systematic ablation experiment configs from a baseline architecture. "
        "Produces variants by: (1) removing one block at a time, (2) swapping the head, "
        "(3) varying d_model. Returns ready-to-use experiment configs. "
        "baseline_blocks: JSON list of block names for the baseline. "
        "baseline_head: head name for the baseline. "
        "ablation_type: 'block_removal', 'head_swap', 'd_model_sweep', or 'all'."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "baseline_blocks": {
                "type": "string",
                "description": "JSON array of block names for baseline architecture",
            },
            "baseline_head": {
                "type": "string",
                "description": "Baseline head name (default: GBMHead)",
            },
            "baseline_d_model": {
                "type": "integer",
                "description": "Baseline d_model (default: 32)",
            },
            "ablation_type": {
                "type": "string",
                "description": (
                    "Type of ablation: 'block_removal' (remove one block at a time), "
                    "'head_swap' (try each head), 'd_model_sweep' (vary hidden dim), "
                    "'block_swap' (substitute each block with alternatives), "
                    "or 'all' (combine all ablation types)"
                ),
            },
            "timeframe": {
                "type": "string",
                "description": "Timeframe for configs: '5m' or '1m' (default: 5m)",
            },
        },
        "required": ["baseline_blocks"],
    },
)
def generate_ablation_configs(
    baseline_blocks: str,
    baseline_head: str = "GBMHead",
    baseline_d_model: int = RESEARCH_D_MODEL,
    ablation_type: str = "all",
    timeframe: str = "5m",
) -> str:
    """Generate ablation experiment configs from a baseline."""
    try:
        blocks = (
            json.loads(baseline_blocks) if isinstance(baseline_blocks, str)
            else baseline_blocks
        )
        configs: list[dict[str, Any]] = []

        # Apply timeframe defaults
        tf_cfg = TIMEFRAME_CONFIGS.get(timeframe, TIMEFRAME_CONFIGS["5m"])
        horizon = int(tf_cfg["pred_len"])
        seq_len = int(tf_cfg["input_len"])

        # Baseline config for reference
        baseline = {
            "name": "baseline",
            "blocks": blocks,
            "head": baseline_head,
            "d_model": baseline_d_model,
            "horizon": horizon,
            "seq_len": seq_len,
            "n_paths": RESEARCH_N_PATHS,
            "lr": RESEARCH_LR,
            "batch_size": RESEARCH_BATCH_SIZE,
            "estimated_params": (
                sum(_estimate_block_params(b, baseline_d_model) for b in blocks)
                + _estimate_head_params(baseline_head, baseline_d_model, horizon)
            ),
        }
        configs.append(baseline)

        # --- Block removal ablation ---
        if ablation_type in ("block_removal", "all"):
            removable = [b for b in blocks if b != "RevIN"]  # Never remove RevIN
            for block in removable:
                ablated = [b for b in blocks if b != block]
                if len(ablated) >= 1:  # Need at least one block
                    params = (
                        sum(_estimate_block_params(b, baseline_d_model) for b in ablated)
                        + _estimate_head_params(
                            baseline_head, baseline_d_model, horizon
                        )
                    )
                    configs.append({
                        "name": f"ablation_remove_{block}",
                        "ablation": f"removed {block}",
                        "blocks": ablated,
                        "head": baseline_head,
                        "d_model": baseline_d_model,
                        "horizon": horizon,
                        "seq_len": seq_len,
                        "n_paths": RESEARCH_N_PATHS,
                        "lr": RESEARCH_LR,
                        "batch_size": RESEARCH_BATCH_SIZE,
                        "estimated_params": params,
                    })

        # --- Head swap ablation ---
        if ablation_type in ("head_swap", "all"):
            all_heads = list(_HEAD_PARAM_FACTORS.keys())
            for head in all_heads:
                if head != baseline_head:
                    params = (
                        sum(_estimate_block_params(b, baseline_d_model) for b in blocks)
                        + _estimate_head_params(head, baseline_d_model, horizon)
                    )
                    configs.append({
                        "name": f"ablation_head_{head}",
                        "ablation": f"swapped head to {head}",
                        "blocks": blocks,
                        "head": head,
                        "d_model": baseline_d_model,
                        "horizon": horizon,
                        "seq_len": seq_len,
                        "n_paths": RESEARCH_N_PATHS,
                        "lr": RESEARCH_LR,
                        "batch_size": RESEARCH_BATCH_SIZE,
                        "estimated_params": params,
                    })

        # --- d_model sweep ---
        if ablation_type in ("d_model_sweep", "all"):
            for dm in [16, 32, 64, 128]:
                if dm != baseline_d_model:
                    params = (
                        sum(_estimate_block_params(b, dm) for b in blocks)
                        + _estimate_head_params(baseline_head, dm, horizon)
                    )
                    configs.append({
                        "name": f"ablation_d_model_{dm}",
                        "ablation": f"d_model={dm} (baseline={baseline_d_model})",
                        "blocks": blocks,
                        "head": baseline_head,
                        "d_model": dm,
                        "horizon": horizon,
                        "seq_len": seq_len,
                        "n_paths": RESEARCH_N_PATHS,
                        "lr": RESEARCH_LR,
                        "batch_size": RESEARCH_BATCH_SIZE,
                        "estimated_params": params,
                    })

        # --- Block swap ablation ---
        if ablation_type in ("block_swap", "all"):
            # For each non-RevIN block, try substituting same-cost alternatives
            _COST_GROUPS: dict[str, list[str]] = {
                "recurrent": ["RNNBlock", "GRUBlock", "LSTMBlock"],
                "conv": ["ResConvBlock", "BiTCNBlock"],
                "attention": ["TransformerBlock", "TransformerEncoder"],
                "special": ["FourierBlock", "TimeMixerBlock", "Unet1DBlock", "TimesNetBlock"],
            }
            swappable = [b for b in blocks if b != "RevIN" and b != "LayerNormBlock"]
            for block in swappable:
                # Find which group this block belongs to
                block_group = None
                for group, members in _COST_GROUPS.items():
                    if block in members:
                        block_group = group
                        break
                if block_group:
                    alternatives = [
                        b for b in _COST_GROUPS[block_group] if b != block
                    ]
                    for alt in alternatives:
                        swapped = [alt if b == block else b for b in blocks]
                        params = (
                            sum(_estimate_block_params(b, baseline_d_model) for b in swapped)
                            + _estimate_head_params(
                                baseline_head, baseline_d_model, horizon
                            )
                        )
                        configs.append({
                            "name": f"ablation_swap_{block}_to_{alt}",
                            "ablation": f"swapped {block} → {alt}",
                            "blocks": swapped,
                            "head": baseline_head,
                            "d_model": baseline_d_model,
                            "horizon": horizon,
                            "seq_len": seq_len,
                            "n_paths": RESEARCH_N_PATHS,
                            "lr": RESEARCH_LR,
                            "batch_size": RESEARCH_BATCH_SIZE,
                            "estimated_params": params,
                        })

        # Sort non-baseline configs by estimated params (cheapest first)
        baseline_cfg = configs[0]
        rest = sorted(configs[1:], key=lambda c: c["estimated_params"])

        return json.dumps({
            "baseline": baseline_cfg,
            "ablations": rest,
            "total_configs": len(configs),
            "note": (
                "Each config is ready for create_experiment(). "
                "Run all on the same deployment for a fair comparison."
            ),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Generate a grid or random sweep of experiment configs over a hyperparameter space. "
        "Specify the architecture (blocks + head) and the ranges to sweep. "
        "Returns ready-to-use experiment configs ranked by estimated cost. "
        "sweep_spec: JSON with 'blocks', 'head', and parameter ranges to sweep."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "blocks": {"type": "string", "description": "JSON array of block names"},
            "head": {"type": "string", "description": "Head name (default: GBMHead)"},
            "d_model_values": {
                "type": "string",
                "description": "JSON array of d_model values to try (e.g. [16, 32, 64])",
            },
            "lr_values": {
                "type": "string",
                "description": "JSON array of learning rates (e.g. [0.0001, 0.001, 0.01])",
            },
            "n_paths_values": {
                "type": "string",
                "description": "JSON array of n_paths values (e.g. [50, 100])",
            },
            "timeframe": {"type": "string", "description": "'5m' or '1m' (default: 5m)"},
            "max_configs": {
                "type": "integer",
                "description": "Max configs to generate (random sample if grid is larger)",
            },
        },
        "required": ["blocks"],
    },
)
def sweep_configs(
    blocks: str,
    head: str = "GBMHead",
    d_model_values: str = "",
    lr_values: str = "",
    n_paths_values: str = "",
    timeframe: str = "5m",
    max_configs: int = 20,
) -> str:
    """Generate a grid/random sweep of experiment configs."""
    try:
        block_list = json.loads(blocks) if isinstance(blocks, str) else blocks
        d_models = (
            json.loads(d_model_values) if isinstance(d_model_values, str) and d_model_values
            else [RESEARCH_D_MODEL]
        )
        lrs = (
            json.loads(lr_values) if isinstance(lr_values, str) and lr_values
            else [RESEARCH_LR]
        )
        n_paths_list = (
            json.loads(n_paths_values) if isinstance(n_paths_values, str) and n_paths_values
            else [RESEARCH_N_PATHS]
        )

        tf_cfg = TIMEFRAME_CONFIGS.get(timeframe, TIMEFRAME_CONFIGS["5m"])
        horizon = int(tf_cfg["pred_len"])
        seq_len = int(tf_cfg["input_len"])

        # Generate full grid
        grid = list(itertools.product(d_models, lrs, n_paths_list))

        # Random sample if grid is too large
        if len(grid) > max_configs:
            grid = random.sample(grid, max_configs)

        configs = []
        for dm, lr, np_ in grid:
            params = (
                sum(_estimate_block_params(b, dm) for b in block_list)
                + _estimate_head_params(head, dm, horizon)
            )
            configs.append({
                "blocks": block_list,
                "head": head,
                "d_model": dm,
                "lr": lr,
                "n_paths": np_,
                "horizon": horizon,
                "seq_len": seq_len,
                "batch_size": RESEARCH_BATCH_SIZE,
                "estimated_params": params,
                "relative_flops": round(
                    _relative_flops(block_list, head, dm, seq_len, horizon)
                ),
            })

        # Sort by cost
        configs.sort(key=lambda c: c["relative_flops"])

        return json.dumps({
            "configs": configs,
            "total_configs": len(configs),
            "grid_size": len(d_models) * len(lrs) * len(n_paths_list),
            "sampled": len(grid) < len(d_models) * len(lrs) * len(n_paths_list),
            "note": (
                "Configs ranked by estimated cost (cheapest first). "
                "Use with create_experiment."
            ),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------------
# Remote proxy tools (low-cost, run on existing deployment)
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Run a fast architecture probe on a Basilica GPU deployment. "
        "Performs a single forward pass + backward pass with random data to check: "
        "(1) shapes are correct, (2) no NaN/Inf in outputs, (3) gradients flow, "
        "(4) initial loss magnitude, (5) gradient norms per layer. "
        "Takes ~2-10 seconds per config vs minutes for full training. "
        "Use this to screen architectures before committing to full training. "
        "deployment_url: URL of a running Basilica deployment. "
        "experiment: experiment config JSON from create_experiment."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "deployment_url": {
                "type": "string",
                "description": "URL of the running Basilica deployment",
            },
            "experiment": {
                "type": "string",
                "description": "Experiment config JSON from create_experiment",
            },
            "timeframe": {
                "type": "string",
                "description": "'5m' or '1m' (default: 5m)",
            },
            "share_token": {
                "type": "string",
                "description": "Optional deployment share token",
            },
        },
        "required": ["deployment_url", "experiment"],
    },
)
def probe_architecture(
    deployment_url: str,
    experiment: str,
    timeframe: str = "5m",
    share_token: str = "",
) -> str:
    """Run a fast forward+backward probe on a deployment."""
    import urllib.error
    import urllib.request

    try:
        exp_dict = json.loads(experiment) if isinstance(experiment, str) else experiment
        exp_dict.pop("timeframe", None)

        tf_cfg = TIMEFRAME_CONFIGS.get(timeframe, TIMEFRAME_CONFIGS["5m"])

        payload = {
            "experiment": exp_dict,
            "timeframe": timeframe,
            "input_len": int(tf_cfg["input_len"]),
            "pred_len": int(tf_cfg["pred_len"]),
        }

        url = deployment_url.rstrip("/")
        if share_token:
            url += f"/probe?token={share_token}"
        else:
            url += "/probe"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        return json.dumps(result, indent=2, default=str)

    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        return json.dumps({
            "error": f"HTTP {exc.code}: {exc.reason}",
            "detail": body,
            "hint": (
                "The /probe endpoint may not be available on this deployment. "
                "Ensure the training server image supports architecture probing."
            ),
        })
    except urllib.error.URLError as exc:
        return json.dumps({
            "error": f"Connection failed: {exc.reason}",
            "hint": "Is the deployment running? Try wait_for_deployment_ready first.",
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Run probes on multiple architecture configs in a single call. "
        "Sends each config to the deployment's /probe endpoint sequentially "
        "and returns a comparison table ranked by initial loss. "
        "Expect ~5-10s per config. Use this to screen candidates before "
        "full training."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "deployment_url": {
                "type": "string",
                "description": "URL of the running Basilica deployment",
            },
            "experiments": {
                "type": "string",
                "description": (
                    "JSON array of experiment configs (from create_experiment or "
                    "generate_ablation_configs)"
                ),
            },
            "timeframe": {
                "type": "string",
                "description": "'5m' or '1m' (default: 5m)",
            },
            "share_token": {
                "type": "string",
                "description": "Optional deployment share token",
            },
        },
        "required": ["deployment_url", "experiments"],
    },
)
def probe_batch(
    deployment_url: str,
    experiments: str,
    timeframe: str = "5m",
    share_token: str = "",
) -> str:
    """Probe multiple architectures and return a ranked comparison."""
    try:
        exp_list = json.loads(experiments) if isinstance(experiments, str) else experiments

        results = []
        for i, exp in enumerate(exp_list):
            exp_json = json.dumps(exp) if isinstance(exp, dict) else exp
            result_str = probe_architecture(
                deployment_url=deployment_url,
                experiment=exp_json,
                timeframe=timeframe,
                share_token=share_token,
            )
            result = json.loads(result_str)
            result["config_index"] = i
            result["config_name"] = (
                exp.get("name", f"config_{i}") if isinstance(exp, dict) else f"config_{i}"
            )
            results.append(result)

        # Separate successes and failures
        successes = [r for r in results if "error" not in r]
        failures = [r for r in results if "error" in r]

        # Rank successes by initial loss (lower = better starting point)
        successes.sort(key=lambda r: r.get("initial_loss", float("inf")))

        return json.dumps({
            "ranked_results": successes,
            "failures": failures,
            "total_probed": len(results),
            "successful": len(successes),
            "failed": len(failures),
            "recommendation": (
                successes[0]["config_name"] if successes else "No successful probes"
            ),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
