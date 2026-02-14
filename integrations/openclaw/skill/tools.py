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

_GET_TIMEOUT = 120   # seconds
_POST_TIMEOUT = 300  # seconds


def _check_curl() -> str | None:
    """Return an error JSON string if curl is not available, else ``None``."""
    if shutil.which("curl") is None:
        return json.dumps({
            "error": "curl not found on PATH. Install curl or use the Python client instead."
        })
    return None


def _curl_get(path: str) -> str:
    """GET request to the bridge, return response body."""
    err = _check_curl()
    if err:
        return err
    try:
        result = subprocess.run(
            ["curl", "-sf", f"{BRIDGE_URL}{path}"],
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
