"""
Hippius storage — decentralised S3-compatible persistence for experiment results.

Hippius provides an S3-compatible API backed by decentralised storage on the
Bittensor network.  We use boto3 to talk to it, so any S3-compatible endpoint
(Cloudflare R2, MinIO, AWS S3) works as a drop-in replacement.

Object layout inside the bucket::

    experiments/{run_id}/{name}.json          individual experiment result
    pipeline_runs/{run_id}/summary.json       full pipeline run summary
    pipeline_runs/{run_id}/comparison.json    CRPS ranking at end of run
    pipeline_runs/latest.json                 pointer to most recent run
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from config import (
    HIPPIUS_ACCESS_KEY,
    HIPPIUS_BUCKET,
    HIPPIUS_ENDPOINT,
    HIPPIUS_SECRET_KEY,
)
from pipeline.tools.registry import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------
_MAX_RETRIES: int = 3
_BACKOFF_BASE: float = 2.0  # seconds: 2, 4, 8

# Set to True after retries are exhausted on a connection error.  Once set,
# all subsequent Hippius calls return immediately instead of blocking for
# minutes with cascading retry cycles (e.g. when running inside Docker
# without network access to the Hippius endpoint).
_endpoint_unreachable: bool = False


def _retry(func, *args, **kwargs):
    """Call *func* with retries and exponential backoff on connection errors."""
    global _endpoint_unreachable
    if _endpoint_unreachable:
        raise ConnectionError("Hippius endpoint previously determined unreachable")
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc)
            # Only retry on transient network / connection errors
            is_transient = any(s in exc_str for s in (
                "Could not connect",
                "EndpointConnectionError",
                "ConnectionError",
                "ConnectTimeoutError",
                "ReadTimeoutError",
                "BrokenPipeError",
            ))
            if not is_transient:
                raise
            delay = _BACKOFF_BASE ** (attempt + 1)
            logger.warning(
                "Hippius request failed (attempt %d/%d), retrying in %.0fs: %s",
                attempt + 1, _MAX_RETRIES, delay, exc,
            )
            time.sleep(delay)
    _endpoint_unreachable = True
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Lazy-loaded boto3 client
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    """Return a boto3 S3 client pointed at the Hippius endpoint."""
    global _client
    if _endpoint_unreachable:
        return None
    if _client is not None:
        return _client

    if not HIPPIUS_ENDPOINT or not HIPPIUS_ACCESS_KEY:
        return None

    import boto3
    from botocore.config import Config as BotoConfig

    _client = boto3.client(
        "s3",
        endpoint_url=HIPPIUS_ENDPOINT,
        aws_access_key_id=HIPPIUS_ACCESS_KEY,
        aws_secret_access_key=HIPPIUS_SECRET_KEY,
        region_name="decentralized",
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    )
    return _client


def _ensure_bucket() -> bool:
    """Create the bucket if it doesn't exist yet.  Returns True if reachable."""
    client = _get_client()
    if client is None:
        return False
    try:
        _retry(client.head_bucket, Bucket=HIPPIUS_BUCKET)
        return True
    except Exception:
        try:
            _retry(client.create_bucket, Bucket=HIPPIUS_BUCKET)
            logger.info("Created Hippius bucket: %s", HIPPIUS_BUCKET)
            return True
        except Exception as exc:
            logger.warning("Could not create bucket %s: %s", HIPPIUS_BUCKET, exc)
            return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _put_json(key: str, data: Any) -> bool:
    """Upload a JSON object.  Returns True on success."""
    client = _get_client()
    if client is None:
        logger.debug("Hippius not configured — skipping upload of %s", key)
        return False
    _ensure_bucket()
    try:
        body = json.dumps(data, indent=2, default=str).encode()
        _retry(
            client.put_object,
            Bucket=HIPPIUS_BUCKET, Key=key, Body=body,
            ContentType="application/json",
        )
        logger.info("Uploaded %s (%d bytes)", key, len(body))
        return True
    except Exception as exc:
        logger.warning("Hippius upload failed for %s: %s", key, exc)
        return False


def _get_json(key: str) -> Any | None:
    """Download and parse a JSON object.  Returns None on failure."""
    client = _get_client()
    if client is None:
        return None
    try:
        resp = _retry(client.get_object, Bucket=HIPPIUS_BUCKET, Key=key)
        return json.loads(resp["Body"].read().decode())
    except client.exceptions.NoSuchKey:
        return None
    except Exception as exc:
        logger.warning("Hippius download failed for %s: %s", key, exc)
        return None


def _list_keys(prefix: str, max_keys: int = 1000) -> list[str]:
    """List object keys under a prefix."""
    client = _get_client()
    if client is None:
        return []
    try:
        resp = _retry(
            client.list_objects_v2,
            Bucket=HIPPIUS_BUCKET, Prefix=prefix, MaxKeys=max_keys,
        )
        return [obj["Key"] for obj in resp.get("Contents", [])]
    except Exception as exc:
        logger.warning("Hippius list failed for prefix %s: %s", prefix, exc)
        return []


# ---------------------------------------------------------------------------
# Run-ID management
# ---------------------------------------------------------------------------
_current_run_id: str | None = None


def get_run_id() -> str:
    """Return (or create) a run ID for the current pipeline invocation."""
    global _current_run_id
    if _current_run_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        _current_run_id = f"{ts}-{uuid.uuid4().hex[:8]}"
    return _current_run_id


def reset_run_id() -> None:
    """Reset so the next pipeline run gets a fresh ID."""
    global _current_run_id
    _current_run_id = None


# ---------------------------------------------------------------------------
# Public save helpers (called by other modules)
# ---------------------------------------------------------------------------

def save_experiment_result(name: str, experiment: dict, result: dict) -> str | None:
    """Persist a single experiment config + result.  Returns the object key or None."""
    run_id = get_run_id()
    key = f"experiments/{run_id}/{name}.json"
    payload = {
        "run_id": run_id,
        "name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "result": result,
    }
    return key if _put_json(key, payload) else None


def save_pipeline_summary(summary: dict) -> str | None:
    """Persist a full pipeline run summary.

    Automatically attaches an ``eval_results`` section that links to the
    individual experiment result objects stored under ``experiments/{run_id}/``.
    This ensures the research summary always references its eval artefacts.
    """
    run_id = get_run_id()
    key = f"pipeline_runs/{run_id}/summary.json"
    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **summary,
    }

    # Attach eval result references so the summary links back to individual results.
    exp_keys = _list_keys(f"experiments/{run_id}/")
    if exp_keys:
        eval_refs = []
        for ek in exp_keys:
            exp_data = _get_json(ek)
            if exp_data and isinstance(exp_data, dict):
                eval_refs.append({
                    "key": ek,
                    "name": exp_data.get("name", ""),
                    "crps": (
                        exp_data.get("result", {}).get("metrics", {}).get("crps")
                        if isinstance(exp_data.get("result"), dict)
                        else None
                    ),
                })
        # Sort by CRPS (best first), nulls last
        eval_refs.sort(
            key=lambda r: r["crps"] if r["crps"] is not None else float("inf")
        )
        payload["eval_results"] = {
            "count": len(eval_refs),
            "experiments": eval_refs,
        }

    ok = _put_json(key, payload)
    if ok:
        # Update the "latest" pointer
        latest = {"run_id": run_id, "timestamp": payload["timestamp"]}
        _put_json("pipeline_runs/latest.json", latest)
    return key if ok else None


def save_comparison(comparison: dict) -> str | None:
    """Persist a comparison / ranking snapshot."""
    run_id = get_run_id()
    key = f"pipeline_runs/{run_id}/comparison.json"
    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "comparison": comparison,
    }
    return key if _put_json(key, payload) else None


# ---------------------------------------------------------------------------
# Agent tools — registered for tool injection
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Save an experiment result to Hippius decentralised storage. "
        "experiment: experiment config JSON. result: run result JSON. "
        "name: optional label for this experiment."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "experiment": {"type": "string", "description": "Experiment config JSON"},
            "result": {"type": "string", "description": "Run result JSON"},
            "name": {
                "type": "string",
                "description": "Experiment label (auto-generated if omitted)",
            },
        },
        "required": ["experiment", "result"],
    },
)
def save_to_hippius(experiment: str, result: str, name: str = "") -> str:
    """Save experiment config + result to Hippius storage."""
    try:
        exp_dict = json.loads(experiment) if isinstance(experiment, str) else experiment
        res_dict = json.loads(result) if isinstance(result, str) else result
        label = name or f"exp-{int(time.time())}"
        key = save_experiment_result(label, exp_dict, res_dict)
        if key:
            return json.dumps({"status": "saved", "key": key, "run_id": get_run_id()})
        return json.dumps({"status": "skipped", "reason": "Hippius not configured"})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "List all pipeline runs stored in Hippius. "
        "Returns run IDs with timestamps, most recent first."
    ),
)
def list_hippius_runs() -> str:
    """List stored pipeline runs."""
    try:
        keys = _list_keys("pipeline_runs/", max_keys=500)
        # Extract unique run IDs from keys like pipeline_runs/{run_id}/summary.json
        runs: dict[str, dict] = {}
        for key in keys:
            parts = key.split("/")
            if len(parts) >= 3 and parts[1] != "latest.json":
                run_id = parts[1]
                if run_id not in runs:
                    runs[run_id] = {"run_id": run_id, "files": []}
                runs[run_id]["files"].append(parts[-1])

        # Sort by run_id (which starts with a timestamp)
        sorted_runs = sorted(runs.values(), key=lambda r: r["run_id"], reverse=True)
        return json.dumps({"runs": sorted_runs, "total": len(sorted_runs)}, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Load a pipeline run summary from Hippius storage. "
        "run_id: the run ID to load (use list_hippius_runs to discover IDs). "
        "Pass 'latest' to load the most recent run."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "run_id": {"type": "string", "description": "Run ID or 'latest'"},
        },
        "required": ["run_id"],
    },
)
def load_hippius_run(run_id: str) -> str:
    """Load a pipeline run from Hippius.

    Returns the summary, comparison, and all individual eval results in a
    single response so callers get the complete research picture.
    """
    try:
        if run_id == "latest":
            latest = _get_json("pipeline_runs/latest.json")
            if not latest:
                return json.dumps({"error": "No runs found in Hippius"})
            run_id = latest["run_id"]

        summary = _get_json(f"pipeline_runs/{run_id}/summary.json")
        comparison = _get_json(f"pipeline_runs/{run_id}/comparison.json")

        data: dict[str, Any] = {"run_id": run_id}
        if summary:
            data["summary"] = summary
        if comparison:
            data["comparison"] = comparison

        # Load individual eval results for this run
        exp_keys = _list_keys(f"experiments/{run_id}/")
        if exp_keys:
            experiments = []
            for key in exp_keys:
                exp = _get_json(key)
                if exp:
                    experiments.append(exp)
            # Sort by CRPS (best first) for easy consumption
            experiments.sort(
                key=lambda e: (
                    e.get("result", {}).get("metrics", {}).get("crps", float("inf"))
                    if isinstance(e.get("result"), dict)
                    else float("inf")
                )
            )
            data["eval_results"] = {
                "count": len(experiments),
                "experiments": experiments,
            }

        if not summary and not comparison and not exp_keys:
            return json.dumps({"error": f"Run {run_id} not found in Hippius"})

        return json.dumps(data, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Load historical experiment results from Hippius across all runs. "
        "Returns experiments sorted by CRPS (best first). "
        "limit: max number of results to return (default 50)."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Max results (default 50)"},
        },
        "required": [],
    },
)
def load_hippius_history(limit: int = 50) -> str:
    """Load all historical experiments from Hippius, ranked by CRPS."""
    try:
        keys = _list_keys("experiments/", max_keys=2000)
        experiments = []
        for key in keys:
            exp = _get_json(key)
            if exp and isinstance(exp, dict):
                result = exp.get("result", {})
                metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
                crps = metrics.get("crps") if isinstance(metrics, dict) else None
                experiments.append({
                    "key": key,
                    "run_id": exp.get("run_id", "unknown"),
                    "name": exp.get("name", "unknown"),
                    "timestamp": exp.get("timestamp", ""),
                    "crps": crps,
                    "metrics": metrics,
                    "blocks": _extract_blocks(exp.get("experiment", {})),
                    "head": _extract_head(exp.get("experiment", {})),
                })

        # Sort by CRPS (lower is better), nulls last
        experiments.sort(key=lambda e: e["crps"] if e["crps"] is not None else float("inf"))
        experiments = experiments[:limit]

        return json.dumps({
            "total_stored": len(keys),
            "returned": len(experiments),
            "experiments": experiments,
        }, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def _extract_blocks(experiment: dict) -> list[str]:
    """Extract block names from an experiment config."""
    try:
        return experiment.get("model", {}).get("backbone", {}).get("blocks", [])
    except (AttributeError, TypeError):
        return []


def _extract_head(experiment: dict) -> str:
    """Extract head name from an experiment config."""
    try:
        target = experiment.get("model", {}).get("head", {}).get("_target_", "")
        return target.split(".")[-1] if target else "unknown"
    except (AttributeError, TypeError):
        return "unknown"
