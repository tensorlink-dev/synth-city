"""
OpenClaw tool definitions for synth-city.

This script is invoked by OpenClaw's workspace skill runner. Each function
maps to a tool that the OpenClaw agent can call during conversations.

OpenClaw workspace skills can execute arbitrary commands. These tools shell
out to ``curl`` against the bridge API so they work without Python dependencies
inside the OpenClaw runtime.

Configuration
-------------
Set the ``SYNTH_BRIDGE_URL`` environment variable to override the default
bridge address (``http://127.0.0.1:8377``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

BRIDGE_URL = os.getenv("SYNTH_BRIDGE_URL", "http://127.0.0.1:8377")
BRIDGE_API_KEY = os.getenv("BRIDGE_API_KEY", "")

_GET_TIMEOUT = 120   # seconds
_POST_TIMEOUT = 300  # seconds


def _check_curl() -> str | None:
    """Return an error JSON string if curl is not available, else ``None``."""
    if shutil.which("curl") is None:
        return json.dumps({
            "error": "curl not found on PATH. Install curl or use the Python client instead."
        })
    return None


def _auth_headers() -> list[str]:
    """Return curl ``-H`` flags for API key authentication, if configured."""
    if not BRIDGE_API_KEY:
        return []
    return ["-H", f"X-API-Key: {BRIDGE_API_KEY}"]


def _curl_get(path: str) -> str:
    """GET request to the bridge, return response body."""
    err = _check_curl()
    if err:
        return err
    try:
        result = subprocess.run(
            ["curl", "-sf", *_auth_headers(), f"{BRIDGE_URL}{path}"],
            capture_output=True,
            text=True,
            timeout=_GET_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({
            "error": f"Request timed out after {_GET_TIMEOUT}s for GET {path}",
        })
    if result.returncode != 0:
        stderr = result.stderr.strip()
        return json.dumps({
            "error": (
                f"Bridge unreachable at {BRIDGE_URL}{path}. "
                f"Is it running? Start with: python main.py bridge"
            ),
            "detail": stderr or f"curl exit code {result.returncode}",
        })
    return result.stdout


def _curl_post(path: str, body: dict) -> str:
    """POST JSON to the bridge, return response body."""
    err = _check_curl()
    if err:
        return err
    try:
        result = subprocess.run(
            [
                "curl", "-sf",
                *_auth_headers(),
                "-X", "POST",
                "-H", "Content-Type: application/json",
                "-d", json.dumps(body),
                f"{BRIDGE_URL}{path}",
            ],
            capture_output=True,
            text=True,
            timeout=_POST_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({
            "error": f"Request timed out after {_POST_TIMEOUT}s for POST {path}",
        })
    if result.returncode != 0:
        stderr = result.stderr.strip()
        return json.dumps({
            "error": (
                f"Bridge unreachable at {BRIDGE_URL}{path}. "
                f"Is it running? Start with: python main.py bridge"
            ),
            "detail": stderr or f"curl exit code {result.returncode}",
        })
    return result.stdout


# ---------------------------------------------------------------------------
# Tools exposed to OpenClaw
# ---------------------------------------------------------------------------

def synth_pipeline_run(publish: bool = False) -> str:
    """Start the synth-city research pipeline to discover and train the best SN50 model."""
    return _curl_post("/pipeline/run", {"publish": publish})


def synth_pipeline_status() -> str:
    """Check the status of a running pipeline."""
    return _curl_get("/pipeline/status")


def synth_list_blocks() -> str:
    """List all available neural network backbone blocks."""
    return _curl_get("/components/blocks")


def synth_list_heads() -> str:
    """List all available prediction head types."""
    return _curl_get("/components/heads")


def synth_list_presets() -> str:
    """List ready-to-run experiment presets."""
    return _curl_get("/components/presets")


def synth_create_experiment(
    blocks: list[str],
    head: str = "GBMHead",
    d_model: int = 32,
    horizon: int = 12,
    n_paths: int = 100,
    lr: float = 0.001,
) -> str:
    """Create a new experiment configuration with blocks, head, and hyperparameters."""
    return _curl_post("/experiment/create", {
        "blocks": blocks,
        "head": head,
        "d_model": d_model,
        "horizon": horizon,
        "n_paths": n_paths,
        "lr": lr,
    })


def synth_validate_experiment(experiment_json: str) -> str:
    """Validate an experiment config without running it. Returns param count and errors."""
    try:
        parsed = json.loads(experiment_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid experiment JSON: {exc}"})
    return _curl_post("/experiment/validate", {"experiment": parsed})


def synth_run_experiment(experiment_json: str, epochs: int = 1, name: str = "") -> str:
    """Run an experiment and return CRPS, sharpness, and log-likelihood metrics."""
    try:
        parsed = json.loads(experiment_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid experiment JSON: {exc}"})
    body: dict = {
        "experiment": parsed,
        "epochs": epochs,
    }
    if name:
        body["name"] = name
    return _curl_post("/experiment/run", body)


def synth_compare_results() -> str:
    """Compare all experiments run in this session, ranked by CRPS score."""
    return _curl_get("/experiment/compare")


def synth_session_summary() -> str:
    """Get a summary of the current research session."""
    return _curl_get("/session/summary")


def synth_get_price(asset: str) -> str:
    """Get the latest price for an SN50 asset (BTC, ETH, SOL, XAU, etc)."""
    return _curl_get(f"/market/price/{asset}")


def synth_get_history(asset: str, days: int = 30) -> str:
    """Get historical price data for an asset."""
    return _curl_get(f"/market/history/{asset}?days={days}")


def synth_clear_session() -> str:
    """Clear the research session and reset accumulated results."""
    return _curl_post("/session/clear", {})


# ---------------------------------------------------------------------------
# Registry / component management
# ---------------------------------------------------------------------------

def synth_list_component_files() -> str:
    """List all component source files in the open-synth-miner registry."""
    return _curl_get("/registry/files")


def synth_read_component(path: str) -> str:
    """Read a component source file to study its structure.

    path: relative to repo root, e.g. 'src/models/components/transformer.py'.
    """
    return _curl_get(f"/registry/read?path={path}")


def synth_write_component(filename: str, code: str) -> str:
    """Write a new PyTorch block or head into the component registry.

    filename: e.g. 'wavelet_block.py'. Written to src/models/components/.
    code: the full Python source code for the component.
    After writing, call synth_reload_registry() to make it available.
    """
    return _curl_post("/registry/write", {"filename": filename, "code": code})


def synth_reload_registry() -> str:
    """Reload the component registry after writing new blocks or heads.

    Call this after synth_write_component() to make new components available
    in synth_list_blocks() and synth_list_heads().
    """
    return _curl_post("/registry/reload", {})


# ---------------------------------------------------------------------------
# HF Hub — model retrieval
# ---------------------------------------------------------------------------

def synth_list_hf_models(repo_id: str = "") -> str:
    """List models published to the Hugging Face Hub repo.

    Returns files, branches, version tags, downloads, and metadata.
    repo_id: HF repo ID (uses default from config if omitted).
    """
    qs = f"?repo_id={repo_id}" if repo_id else ""
    return _curl_get(f"/hf/models{qs}")


def synth_fetch_hf_model_card(repo_id: str = "", revision: str = "main") -> str:
    """Fetch the model card (README) and config from a HF Hub repo.

    Returns card content, structured metadata, and config.json if present.
    """
    parts = [f"revision={revision}"]
    if repo_id:
        parts.append(f"repo_id={repo_id}")
    return _curl_get(f"/hf/model-card?{'&'.join(parts)}")


def synth_fetch_hf_artifact(
    filename: str, repo_id: str = "", revision: str = "main",
) -> str:
    """Download a JSON artifact from the HF Hub repo.

    filename: path within the repo (e.g. 'experiment.json', 'metrics.json').
    """
    parts = [f"filename={filename}", f"revision={revision}"]
    if repo_id:
        parts.append(f"repo_id={repo_id}")
    return _curl_get(f"/hf/artifact?{'&'.join(parts)}")


# ---------------------------------------------------------------------------
# History / tested models
# ---------------------------------------------------------------------------

def synth_list_history_runs() -> str:
    """List all pipeline runs stored in Hippius decentralised storage.

    Returns run IDs with timestamps, most recent first.
    """
    return _curl_get("/history/runs")


def synth_load_history_run(run_id: str) -> str:
    """Load a specific pipeline run from Hippius storage.

    run_id: the run ID (from synth_list_history_runs), or 'latest'.
    Returns the summary, comparison ranking, and individual experiments.
    """
    return _curl_get(f"/history/run/{run_id}")


def synth_load_tested_experiments(limit: int = 50) -> str:
    """Load the best tested experiments across all pipeline runs.

    Returns experiments sorted by CRPS (best first) with their block
    compositions, heads, and metrics. Use this to see which architectures
    have already been tried and how they performed.
    """
    return _curl_get(f"/history/experiments?limit={limit}")


def synth_fetch_wandb_runs(limit: int = 20, order: str = "best") -> str:
    """Fetch experiment runs from Weights & Biases.

    Returns run names, configs, CRPS scores, and W&B URLs.
    order: 'best' (lowest CRPS first), 'recent' (newest first), or 'worst'.
    """
    return _curl_get(f"/history/wandb?limit={limit}&order={order}")


# ---------------------------------------------------------------------------
# CLI entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools.py <tool_name> [args...]")
        print(f"Bridge URL: {BRIDGE_URL}")
        print("\nAvailable tools:")
        for name in sorted(dir()):
            if name.startswith("synth_"):
                print(f"  {name}")
        sys.exit(0)

    tool_name = sys.argv[1]
    func = globals().get(tool_name)
    if func is None or not callable(func):
        print(f"Unknown tool: {tool_name}")
        sys.exit(1)

    args = sys.argv[2:]
    result = func(*args)
    print(result)
