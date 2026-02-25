"""Async training server for Basilica GPU deployments.

Runs inside a Docker container on a Basilica GPU pod.  Accepts experiment
configs via ``POST /train`` and returns metrics (including CRPS) as an
NDJSON stream with periodic heartbeats to keep connections alive through
Basilica's reverse proxy and load-balancer infrastructure.

Endpoints
---------
GET  /                → ``{"ok": true}``  (Basilica reverse-proxy readiness)
HEAD /                → 200 OK (Basilica ``HEAD /`` probe)
GET  /health          → CUDA + ResearchSession diagnostics
POST /train           → NDJSON stream: heartbeats + final result
GET  /gpu             → ``nvidia-smi`` output for diagnostics
GET  /job-result/{job_id} → recover result after stream drop
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import subprocess
import time
import traceback
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("training_server")

app = FastAPI(title="synth-city Basilica Training Server")

# ---------------------------------------------------------------------------
# Heartbeat / streaming config
# ---------------------------------------------------------------------------

# Interval (seconds) between heartbeat lines in the NDJSON stream.
# Cloud reverse proxies typically drop idle connections after 60-300s.
# 15s keeps the connection alive with wide headroom.
STREAM_HEARTBEAT_INTERVAL_SEC: float = 15.0

# Padding so the chunk exceeds typical reverse-proxy buffer thresholds
# (commonly 4-8 KB) and gets flushed immediately.
_HEARTBEAT_PAD = " " * 8192

# ---------------------------------------------------------------------------
# In-memory job result cache
#
# When a streaming response is severed by a reverse proxy (e.g. Basilica's
# ingress enforcing a connection-time limit), the final NDJSON result line
# never reaches the client.  The server caches completed results here so
# the client can recover via GET /job-result/{job_id}.
# ---------------------------------------------------------------------------

_JOB_RESULTS: dict[str, tuple[float, dict]] = {}  # job_id -> (timestamp, result)
_JOB_RESULT_TTL_SEC: float = float(os.environ.get("JOB_RESULT_TTL_SEC", "14400"))  # 4h

_ACTIVE_JOBS: dict[str, float] = {}  # job_id -> start_timestamp

# Strong-reference set for background tasks so they don't get GC'd.
_BACKGROUND_TASKS: set[asyncio.Task] = set()


def _mark_job_active(job_id: str) -> None:
    _ACTIVE_JOBS[job_id] = time.time()


def _mark_job_inactive(job_id: str) -> None:
    _ACTIVE_JOBS.pop(job_id, None)


def _is_job_active(job_id: str) -> bool:
    return job_id in _ACTIVE_JOBS


def _evict_stale_results() -> None:
    cutoff = time.time() - _JOB_RESULT_TTL_SEC
    stale = [k for k, (ts, _) in _JOB_RESULTS.items() if ts < cutoff]
    for k in stale:
        _JOB_RESULTS.pop(k, None)


def _cache_job_result(job_id: str, result: dict) -> None:
    _evict_stale_results()
    _JOB_RESULTS[job_id] = (time.time(), result)
    _mark_job_inactive(job_id)


def _get_job_result(job_id: str) -> dict | None:
    _evict_stale_results()
    entry = _JOB_RESULTS.get(job_id)
    if entry is None:
        return None
    ts, result = entry
    if ts < time.time() - _JOB_RESULT_TTL_SEC:
        _JOB_RESULTS.pop(job_id, None)
        return None
    return result


# ---------------------------------------------------------------------------
# Build version — baked in at Docker build time via --build-arg or read from
# a file written by CI.
# ---------------------------------------------------------------------------

_BUILD_VERSION: str | None = None
_BUILD_VERSION_INITIALIZED: bool = False


def _get_build_version() -> str | None:
    global _BUILD_VERSION, _BUILD_VERSION_INITIALIZED
    if _BUILD_VERSION_INITIALIZED:
        return _BUILD_VERSION
    ver = os.getenv("BUILD_VERSION", "").strip()
    if ver:
        _BUILD_VERSION = ver
        _BUILD_VERSION_INITIALIZED = True
        return ver
    for path in ("/app/.build_version", ".build_version"):
        try:
            with open(path, encoding="utf-8") as f:
                ver = f.read().strip()
                if ver:
                    _BUILD_VERSION = ver
                    _BUILD_VERSION_INITIALIZED = True
                    return ver
        except (FileNotFoundError, OSError):
            pass
    _BUILD_VERSION_INITIALIZED = True
    return None


# ---------------------------------------------------------------------------
# CUDA diagnostics (cached at module level)
# ---------------------------------------------------------------------------

_CUDA_INFO: dict[str, Any] | None = None


def _detect_cuda_info() -> dict[str, Any]:
    """Detect CUDA availability and verify it actually works."""
    info: dict[str, Any] = {}

    # nvcc version
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True, timeout=5)
        for line in out.splitlines():
            if "release" in line.lower():
                parts = line.split("release")[-1].strip().rstrip(".")
                info["cuda_toolkit"] = parts.split(",")[0].strip()
                break
    except Exception:
        pass

    # PyTorch CUDA
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda_version"] = torch.version.cuda or "none"
        info["cuda_available"] = torch.cuda.is_available()

        if not torch.cuda.is_available():
            return info

        dev_idx = torch.cuda.current_device()
        info["cuda_device"] = torch.cuda.get_device_name(dev_idx)

        # GPU memory
        try:
            _free, total = torch.cuda.mem_get_info(dev_idx)
            info["gpu_total_memory_gb"] = round(total / (1024**3), 1)
        except Exception:
            pass

        # Functional check: verify tensor ops actually work on GPU
        try:
            device = torch.device(f"cuda:{dev_idx}")
            t = torch.tensor([1.0, 2.0], device=device)
            result = float((t * t).sum().item())
            info["cuda_functional"] = abs(result - 5.0) < 1e-5
            del t
        except Exception as exc:
            info["cuda_functional"] = False
            info["cuda_functional_warning"] = str(exc)

    except Exception as exc:
        info["torch_error"] = str(exc)

    return info


def _get_cuda_info() -> dict[str, Any]:
    global _CUDA_INFO
    if _CUDA_INFO is None:
        _CUDA_INFO = _detect_cuda_info()
    return _CUDA_INFO


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
            logger.warning("Failed to import %s: %s: %s", mod_path, type(exc).__name__, exc)
            errors.append((mod_path, exc))
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
# Synchronous training runner (runs in thread pool)
# ---------------------------------------------------------------------------


def _run_training_sync(
    experiment: dict,
    epochs: int,
    hf_repo: str,
    asset_files: dict[str, str],
    input_len: int,
    pred_len: int,
) -> dict:
    """Execute training synchronously.  Called from a thread via asyncio."""
    session_cls = _get_session_class()
    session = session_cls()

    loader = None
    if hf_repo and asset_files:
        loader = _build_data_loader(hf_repo, asset_files, input_len, pred_len)

    run_kwargs: dict = {"epochs": epochs}
    if loader is not None:
        run_kwargs["data_loader"] = loader

    try:
        result = session.run(experiment, **run_kwargs)
    except TypeError:
        run_kwargs.pop("data_loader", None)
        logger.warning("session.run() rejected data_loader kwarg, retrying without")
        result = session.run(experiment, **run_kwargs)

    return result


# ---------------------------------------------------------------------------
# Background task helpers
# ---------------------------------------------------------------------------


def _log_background_task_exception(t: asyncio.Task) -> None:
    if t.cancelled():
        return
    exc = t.exception()
    if exc is not None:
        logger.error("Background _wait_and_persist task failed: %s", exc, exc_info=exc)


async def _wait_and_persist(
    task: asyncio.Task,
    job_id: str,
    t0: float,
    timeout: float,
) -> None:
    """Background: wait for training to complete after stream drop, then cache result."""
    train_result = None
    train_error: Exception | None = None

    try:
        train_result = await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        train_error = TimeoutError(
            f"Training timed out after {timeout:.0f}s (background wait after stream drop)"
        )
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    except asyncio.CancelledError:
        train_error = RuntimeError("Training task was cancelled")
    except Exception as exc:
        train_error = exc

    # If task finished but we didn't capture the result above
    if train_error is None and train_result is None and task.done() and not task.cancelled():
        exc = task.exception()
        if exc is not None:
            train_error = exc
        else:
            train_result = task.result()

    elapsed = time.time() - t0

    if train_error is not None:
        _cache_job_result(job_id, {
            "type": "error",
            "error": str(train_error),
            "elapsed_sec": round(elapsed, 1),
            "job_id": job_id,
        })
    elif train_result is not None:
        _cache_job_result(job_id, {
            "type": "result",
            "job_id": job_id,
            **_sanitize_result(train_result),
        })
    else:
        _cache_job_result(job_id, {
            "type": "error",
            "error": "Training cancelled before completion",
            "elapsed_sec": round(elapsed, 1),
            "job_id": job_id,
        })


def _sanitize_result(result: Any) -> dict:
    """Ensure a training result is JSON-serializable."""
    if isinstance(result, dict):
        try:
            json.dumps(result)
            return result
        except (TypeError, ValueError):
            return json.loads(json.dumps(result, default=str))
    return {"raw": str(result)}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
@app.head("/")
async def root():
    """Root endpoint for Basilica reverse-proxy readiness probes.

    Basilica's reverse proxy sends ``HEAD /`` to determine if the container
    is ready to receive traffic.  Returning 200 here ensures the pod is
    marked healthy and external requests are routed to this container.
    """
    return {"ok": True}


@app.get("/health")
async def health():
    """Health check with CUDA diagnostics.

    Kept cheap (no network calls) so infrastructure probes never timeout.
    """
    try:
        _get_session_class()
        session_ok = True
        session_error = None
    except Exception as exc:
        session_ok = False
        session_error = str(exc)

    cuda_info = _get_cuda_info()

    resp: dict[str, Any] = {
        "status": "ok" if session_ok else "error",
        "ts": time.time(),
        "research_session": session_ok,
        "cuda": cuda_info,
    }

    if session_error:
        resp["error"] = session_error

    build_ver = _get_build_version()
    if build_ver:
        resp["build_version"] = build_ver

    if not session_ok:
        return JSONResponse(resp, status_code=503)
    return resp


@app.get("/gpu")
async def gpu_info():
    """Return nvidia-smi output."""
    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "nvidia_smi": proc.stdout,
            "returncode": proc.returncode,
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/job-result/{job_id}")
async def get_job_result_endpoint(job_id: str):
    """Recover a training result after a stream drop.

    When a ``POST /train`` streaming response is severed mid-flight
    (e.g. by a reverse-proxy connection-time limit), the result is lost
    in transit even though training may have completed.  This endpoint
    lets the client recover the result from the server's in-memory cache.

    Returns:
        200 with the result dict if found.
        202 if the job is still actively training.
        404 if the job is unknown.
    """
    result = _get_job_result(job_id)
    if result is not None:
        return result

    if _is_job_active(job_id):
        started = _ACTIVE_JOBS.get(job_id, 0.0)
        elapsed = time.time() - started if started else 0.0
        return JSONResponse(
            status_code=202,
            content={
                "status": "training",
                "job_id": job_id,
                "elapsed_sec": round(elapsed, 1),
                "detail": "Training is still in progress",
            },
        )

    raise HTTPException(status_code=404, detail=f"No cached result for job {job_id}")


@app.post("/train")
async def train(request: Request):
    """Run an experiment and return results as an NDJSON stream.

    Request JSON::

        {
            "experiment": { ... },
            "epochs": 1,
            "timeframe": "5m",
            "hf_repo": "...",
            "asset_files": { ... },
            "input_len": 288,
            "pred_len": 288,
            "job_id": "..."          # optional, auto-generated if absent
        }

    Response: NDJSON stream (``application/x-ndjson``) with:
    - **heartbeat** lines every ~15s to keep connection alive
    - A final **result** or **error** line

    Clients that don't need streaming can read the *last* line — it is
    always the result or error.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    experiment = body.get("experiment")
    if not experiment:
        raise HTTPException(status_code=400, detail="Missing 'experiment' field")

    epochs = body.get("epochs", 1)
    hf_repo = body.get("hf_repo", "")
    asset_files = body.get("asset_files", {})
    input_len = body.get("input_len", 288)
    pred_len = body.get("pred_len", 288)
    job_id = body.get("job_id") or uuid.uuid4().hex

    # Strip timeframe tag (used by caller, not by ResearchSession)
    experiment.pop("timeframe", None)

    _mark_job_active(job_id)

    async def _stream():
        t0 = time.time()

        # Initial heartbeat — flush immediately so the reverse proxy begins
        # forwarding the chunked response.
        init_hb = json.dumps({
            "type": "heartbeat",
            "job_id": job_id,
            "elapsed_sec": 0.0,
            "_pad": _HEARTBEAT_PAD,
        })
        yield init_hb.encode("utf-8") + b"\n"

        # Run the synchronous training in a thread so the event loop stays
        # responsive for heartbeats and health probes.
        task = asyncio.create_task(
            asyncio.to_thread(
                _run_training_sync,
                experiment,
                epochs,
                hf_repo,
                asset_files,
                input_len,
                pred_len,
            )
        )

        try:
            # Emit heartbeats until training completes.
            while not task.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(task),
                        timeout=STREAM_HEARTBEAT_INTERVAL_SEC,
                    )
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    break

                elapsed = time.time() - t0
                hb = json.dumps({
                    "type": "heartbeat",
                    "job_id": job_id,
                    "elapsed_sec": round(elapsed, 1),
                    "_pad": _HEARTBEAT_PAD,
                })
                yield hb.encode("utf-8") + b"\n"

        finally:
            # Always persist the result for recovery, even if the stream
            # is severed (GeneratorExit).
            elapsed = time.time() - t0

            if task.done():
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

                if not task.cancelled():
                    exc = task.exception()
                    if exc is not None:
                        _cache_job_result(job_id, {
                            "type": "error",
                            "error": str(exc),
                            "traceback": "".join(
                                traceback.format_exception(type(exc), exc, exc.__traceback__)
                            ),
                            "elapsed_sec": round(elapsed, 1),
                            "job_id": job_id,
                        })
                    else:
                        result = task.result()
                        _cache_job_result(job_id, {
                            "type": "result",
                            "job_id": job_id,
                            **_sanitize_result(result),
                        })
                else:
                    _cache_job_result(job_id, {
                        "type": "error",
                        "error": "Training cancelled",
                        "elapsed_sec": round(elapsed, 1),
                        "job_id": job_id,
                    })
            else:
                # Training still running — continue in background so we
                # don't waste GPU work just because the proxy killed the
                # HTTP connection.
                remaining = max(7200 - elapsed + 60, 60.0)
                logger.warning(
                    "Stream severed for job %s while training is still "
                    "in progress (%.0fs elapsed). Training continues in "
                    "background — result recoverable via GET /job-result/%s",
                    job_id, elapsed, job_id,
                )
                bg = asyncio.create_task(
                    _wait_and_persist(task, job_id, t0, remaining)
                )
                _BACKGROUND_TASKS.add(bg)
                bg.add_done_callback(_BACKGROUND_TASKS.discard)
                bg.add_done_callback(_log_background_task_exception)

        # Yield the final result/error line (only reached if stream
        # wasn't severed).
        payload = _get_job_result(job_id)
        if payload is not None:
            try:
                yield json.dumps(payload, default=str).encode("utf-8") + b"\n"
            except Exception as exc:
                yield json.dumps({
                    "type": "error",
                    "error": f"Result serialization failed: {exc}",
                    "job_id": job_id,
                }).encode("utf-8") + b"\n"

    return StreamingResponse(
        _stream(),
        media_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache, no-transform",
            "X-Content-Type-Options": "nosniff",
        },
    )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def _startup():
    port = os.environ.get("TRAINING_SERVER_PORT", "8378")
    logger.info("Training server starting on port %s", port)

    # Eagerly import ResearchSession so errors appear in pod logs immediately.
    try:
        _get_session_class()
        logger.info("Startup self-test passed: ResearchSession is importable")
    except Exception as exc:
        logger.error(
            "Startup self-test FAILED: ResearchSession is NOT importable. "
            "/health will return 503. Error: %s",
            exc,
            exc_info=True,
        )

    # Eagerly detect CUDA info and log it.
    cuda_info = _get_cuda_info()
    if cuda_info.get("cuda_available"):
        logger.info(
            "CUDA: %s, %s GB, functional=%s",
            cuda_info.get("cuda_device", "?"),
            cuda_info.get("gpu_total_memory_gb", "?"),
            cuda_info.get("cuda_functional", "?"),
        )
    else:
        logger.warning("CUDA is NOT available — training will likely fail")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("TRAINING_SERVER_PORT", "8378"))
    uvicorn.run(app, host="0.0.0.0", port=port)
