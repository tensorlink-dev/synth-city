"""Lightweight HTTP training server for Basilica GPU deployments.

Runs inside a Docker container on a Basilica GPU pod.  Accepts experiment
configs via ``POST /train`` and returns metrics (including CRPS) as JSON.

Endpoints
---------
GET  /health          → ``{"status": "ok"}``
POST /train           → run an experiment and return results
GET  /gpu             → ``nvidia-smi`` output for diagnostics
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import traceback

# Early stdout print before heavy imports — if this doesn't appear in
# container logs, Python itself is crashing (segfault, OOM, bad image).
print("training_server.py: stdlib imports OK, loading flask...", flush=True)

from flask import Flask, Response, jsonify, request  # noqa: E402

print("training_server.py: flask OK", flush=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("training_server")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# ResearchSession import (cached)
# ---------------------------------------------------------------------------

_session_cls = None


def _get_session_class():
    global _session_cls
    if _session_cls is not None:
        return _session_cls
    errors: list[tuple[str, Exception]] = []
    for mod_path in ("osa.research.agent_api", "src.research.agent_api", "research.agent_api"):
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, "ResearchSession", None)
            if cls is not None:
                _session_cls = cls
                logger.info("Loaded ResearchSession from %s", mod_path)
                return cls
        except Exception as exc:
            # Catch ALL exceptions (not just ImportError) so that runtime
            # failures in transitive imports (torch CUDA init, missing .so
            # files, etc.) don't prevent trying the fallback module path.
            logger.warning("Failed to import %s: %s: %s", mod_path, type(exc).__name__, exc)
            errors.append((mod_path, exc))
    # Build a detailed error message listing what failed for each path
    details = "; ".join(f"{path}: {type(exc).__name__}: {exc}" for path, exc in errors)
    raise ImportError(
        f"Cannot import ResearchSession from any known module path. "
        f"Errors: {details}"
    )


# ---------------------------------------------------------------------------
# Data loader helper
# ---------------------------------------------------------------------------

def _build_data_loader(
    hf_repo: str,
    asset_files: dict[str, str],
    input_len: int,
    pred_len: int,
):
    """Build a MarketDataLoader that fetches data from HuggingFace."""
    try:
        from osa.data.market_data_loader import (
            HFOHLCVSource,
            MarketDataLoader,
            ZScoreEngineer,
        )
        source = HFOHLCVSource(
            repo_id=hf_repo,
            asset_files=asset_files,
            repo_type="dataset",
        )
        return MarketDataLoader(
            data_source=source,
            engineer=ZScoreEngineer(),
            assets=list(asset_files.keys()),
            input_len=input_len,
            pred_len=pred_len,
            batch_size=64,
            feature_dim=4,
            gap_handling="ffill",
            stride=12,
        )
    except Exception as exc:
        logger.warning("Could not build data loader: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(Exception)
def handle_exception(exc):
    """Catch-all: ensure unhandled exceptions return JSON, not Flask HTML."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    try:
        return jsonify({
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }), 500
    except Exception:
        return Response(
            json.dumps({"status": "error", "error": str(exc)}),
            status=500,
            content_type="application/json",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Health check — verifies ResearchSession is importable."""
    try:
        _get_session_class()
        return jsonify({"status": "ok"})
    except ImportError as exc:
        return jsonify({"status": "error", "error": str(exc)}), 503
    except Exception as exc:
        logger.error("Health check failed: %s", exc, exc_info=True)
        # Use Response+json.dumps instead of jsonify as a safety net —
        # if the exception contains non-serialisable data, jsonify can
        # itself raise, causing Flask to return an empty-body 500.
        return Response(
            json.dumps({
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }),
            status=500,
            content_type="application/json",
        )


@app.route("/gpu", methods=["GET"])
def gpu_info():
    """Return nvidia-smi output."""
    try:
        proc = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10,
        )
        return jsonify({
            "nvidia_smi": proc.stdout,
            "returncode": proc.returncode,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/train", methods=["POST"])
def train():
    """Run an experiment and return results.

    Request JSON::

        {
            "experiment": { ... },       # experiment config from create_experiment
            "epochs": 1,                 # training epochs (default: 1)
            "timeframe": "5m",           # "5m" or "1m" (default: "5m")
            "hf_repo": "...",            # HF dataset repo
            "asset_files": { ... },      # asset → file path mapping
            "input_len": 288,
            "pred_len": 288
        }

    Response JSON: the result dict from ``ResearchSession.run()``.
    """
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"status": "error", "error": "Invalid JSON body"}), 400

    experiment = body.get("experiment")
    if not experiment:
        return jsonify({"status": "error", "error": "Missing 'experiment' field"}), 400

    epochs = body.get("epochs", 1)
    hf_repo = body.get("hf_repo", "")
    asset_files = body.get("asset_files", {})
    input_len = body.get("input_len", 288)
    pred_len = body.get("pred_len", 288)

    # Strip timeframe tag (used by caller, not by ResearchSession)
    experiment.pop("timeframe", None)

    try:
        session_cls = _get_session_class()
        session = session_cls()

        # Build data loader if HF info was provided
        loader = None
        if hf_repo and asset_files:
            loader = _build_data_loader(hf_repo, asset_files, input_len, pred_len)

        # Run the experiment
        run_kwargs: dict = {"epochs": epochs}
        if loader is not None:
            run_kwargs["data_loader"] = loader

        try:
            result = session.run(experiment, **run_kwargs)
        except TypeError:
            # ResearchSession.run() may not accept data_loader yet
            run_kwargs.pop("data_loader", None)
            logger.warning("session.run() rejected data_loader kwarg, retrying without")
            result = session.run(experiment, **run_kwargs)

        # Use default=str to handle numpy floats, datetime, torch tensors, etc.
        return Response(
            json.dumps(result, default=str),
            content_type="application/json",
        )

    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("TRAINING_SERVER_PORT", "8378"))
    print(f"training_server.py: starting on port {port}", flush=True)
    logger.info("Starting training server on port %d", port)

    # Print environment diagnostics for debugging container issues
    print("training_server.py: environment diagnostics:", flush=True)
    try:
        import torch
        print(f"  torch={torch.__version__}, cuda={torch.cuda.is_available()}", flush=True)
    except Exception as exc:
        print(f"  torch import FAILED: {exc}", flush=True)

    # Eagerly try to import ResearchSession at startup so any errors
    # appear in the pod logs immediately instead of only on first request.
    try:
        _get_session_class()
        logger.info("Startup self-test passed: ResearchSession is importable")
        print("training_server.py: ResearchSession import OK", flush=True)
    except Exception as exc:
        logger.error(
            "Startup self-test FAILED: ResearchSession is NOT importable. "
            "/health will return 503. Error: %s",
            exc,
            exc_info=True,
        )
        print(f"training_server.py: ResearchSession import FAILED: {exc}", flush=True)

    print(f"training_server.py: starting Flask on 0.0.0.0:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, threaded=False)
