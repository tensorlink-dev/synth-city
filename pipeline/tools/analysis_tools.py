"""
Analysis tools — read back from Hippius and HF Hub for historical experiment analysis.

These tools let agents query past results from persistent storage:
  - Hippius: fetch runs, compare CRPS trends, find best historical configs
  - HF Hub: list published models, read model cards, compare versions
"""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any

from config import HF_REPO_ID, HF_TOKEN
from pipeline.tools.registry import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hippius-backed experiment analysis tools
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Fetch experiment runs from Hippius decentralised storage. "
        "Returns run names, configs, metrics, and CRPS scores. "
        "limit: max runs to return (default 20). "
        "order: sort order — 'best' (lowest CRPS first), 'recent' (newest first), or 'worst'."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Max runs to return (default 20)"},
            "order": {
                "type": "string",
                "description": "'best', 'recent', or 'worst' (default 'best')",
            },
        },
        "required": [],
    },
)
def fetch_experiment_runs(limit: int = 20, order: str = "best") -> str:
    """Fetch past experiment runs from Hippius storage."""
    try:
        from pipeline.tools.hippius_store import _get_json, _list_keys

        keys = _list_keys("experiments/", max_keys=2000)

        experiments = []
        for key in keys:
            exp = _get_json(key)
            if exp and isinstance(exp, dict):
                result = exp.get("result", {})
                metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
                crps = metrics.get("crps") if isinstance(metrics, dict) else None
                experiments.append({
                    "id": exp.get("run_id", "unknown"),
                    "name": exp.get("name", "unknown"),
                    "created_at": exp.get("timestamp", ""),
                    "config": exp.get("experiment", {}),
                    "metrics": metrics,
                    "crps": crps,
                })

        # Sort based on order
        if order == "best":
            experiments.sort(
                key=lambda e: e["crps"] if e["crps"] is not None else float("inf")
            )
        elif order == "worst":
            experiments.sort(
                key=lambda e: e["crps"] if e["crps"] is not None else float("-inf"),
                reverse=True,
            )
        else:  # recent
            experiments.sort(key=lambda e: e["created_at"], reverse=True)

        experiments = experiments[:limit]

        return json.dumps(
            {"total": len(experiments), "order": order, "runs": experiments},
            indent=2, default=str,
        )
    except Exception as exc:
        return json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


@tool(
    description=(
        "Get detailed information about a specific pipeline run from Hippius storage. "
        "run_id: the pipeline run ID (from fetch_experiment_runs or list_hippius_runs). "
        "Pass 'latest' to load the most recent run."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "run_id": {"type": "string", "description": "Pipeline run ID or 'latest'"},
        },
        "required": ["run_id"],
    },
)
def get_experiment_run_detail(run_id: str) -> str:
    """Get full details for a specific pipeline run from Hippius."""
    try:
        from pipeline.tools.hippius_store import _get_json, _list_keys

        if run_id == "latest":
            latest = _get_json("pipeline_runs/latest.json")
            if not latest:
                return json.dumps({"error": "No runs found in Hippius"})
            run_id = latest["run_id"]

        summary = _get_json(f"pipeline_runs/{run_id}/summary.json")

        data: dict[str, Any] = {"run_id": run_id}
        if summary:
            data["name"] = summary.get("name", run_id)
            data["created_at"] = summary.get("timestamp", "")
            data["config"] = summary.get("experiment", summary.get("config", {}))
            data["metrics"] = summary.get("metrics", {})

        # Load individual eval results for this run
        exp_keys = _list_keys(f"experiments/{run_id}/")
        experiments = []
        for key in exp_keys:
            exp = _get_json(key)
            if exp and isinstance(exp, dict):
                result = exp.get("result", {})
                metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
                experiments.append({
                    "name": exp.get("name", "unknown"),
                    "timestamp": exp.get("timestamp", ""),
                    "config": exp.get("experiment", {}),
                    "metrics": metrics,
                    "crps": metrics.get("crps") if isinstance(metrics, dict) else None,
                })

        # Sort by CRPS (best first)
        experiments.sort(
            key=lambda e: e["crps"] if e["crps"] is not None else float("inf")
        )
        data["experiments"] = experiments
        data["experiment_count"] = len(experiments)

        if not summary and not experiments:
            return json.dumps({"error": f"Run {run_id} not found in Hippius"})

        return json.dumps(data, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Analyse CRPS trends across experiment runs over time. "
        "Returns a time-ordered series of best CRPS scores showing improvement trajectory. "
        "limit: number of recent runs to analyse (default 50)."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Number of runs to analyse (default 50)"},
        },
        "required": [],
    },
)
def analyze_experiment_trends(limit: int = 50) -> str:
    """Analyse CRPS improvement trends across experiments in Hippius."""
    try:
        from pipeline.tools.hippius_store import _get_json, _list_keys

        keys = _list_keys("experiments/", max_keys=2000)

        entries = []
        for key in keys:
            exp = _get_json(key)
            if not exp or not isinstance(exp, dict):
                continue
            result = exp.get("result", {})
            metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
            crps = metrics.get("crps") if isinstance(metrics, dict) else None
            if crps is not None:
                entries.append({
                    "run_id": exp.get("run_id", "unknown"),
                    "name": exp.get("name", "unknown"),
                    "created_at": exp.get("timestamp", ""),
                    "crps": crps,
                    "sharpness": metrics.get("sharpness"),
                    "log_likelihood": metrics.get("log_likelihood"),
                })

        if not entries:
            return json.dumps({"error": "No experiments with CRPS found in Hippius"})

        # Sort chronologically
        entries.sort(key=lambda e: e["created_at"])

        # Limit to the most recent N entries
        entries = entries[-limit:]

        # Compute running best
        running_best = float("inf")
        for entry in entries:
            if entry["crps"] < running_best:
                running_best = entry["crps"]
            entry["running_best_crps"] = running_best

        best_entry = min(entries, key=lambda e: e["crps"])

        return json.dumps({
            "total_runs": len(entries),
            "best_crps": best_entry["crps"],
            "best_run": best_entry["name"],
            "latest_crps": entries[-1]["crps"] if entries else None,
            "improvement": entries[0]["crps"] - best_entry["crps"] if len(entries) > 1 else 0,
            "timeline": entries,
        }, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------------
# HF Hub analysis tools
# ---------------------------------------------------------------------------

@tool(
    description=(
        "List models published to the Hugging Face Hub repository. "
        "Returns model versions with metadata, tags, and download counts. "
        "repo_id: HF repo (defaults to config HF_REPO_ID)."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "repo_id": {"type": "string", "description": "HF repo ID (default from config)"},
        },
        "required": [],
    },
)
def list_hf_models(repo_id: str = "") -> str:
    """List model files and revisions in the HF Hub repo."""
    try:
        from huggingface_hub import HfApi, list_repo_refs

        target_repo = repo_id or HF_REPO_ID
        if not target_repo:
            return json.dumps({"error": "No HF_REPO_ID configured"})

        api = HfApi(token=HF_TOKEN or None)

        # Get repo info
        info = api.repo_info(repo_id=target_repo, repo_type="model")

        # List all files in main branch
        files = api.list_repo_tree(repo_id=target_repo, repo_type="model")
        file_list = []
        for f in files:
            entry = {"path": f.rfilename if hasattr(f, "rfilename") else str(f)}
            if hasattr(f, "size"):
                entry["size_bytes"] = f.size
            file_list.append(entry)

        # List branches/tags (model versions)
        refs = list_repo_refs(repo_id=target_repo, repo_type="model", token=HF_TOKEN or None)
        branches = [{"name": b.name, "ref": b.ref} for b in refs.branches] if refs.branches else []
        tags = [{"name": t.name, "ref": t.ref} for t in refs.tags] if refs.tags else []

        result: dict[str, Any] = {
            "repo_id": target_repo,
            "last_modified": str(info.last_modified) if info.last_modified else None,
            "downloads": info.downloads,
            "likes": info.likes,
            "tags": list(info.tags) if info.tags else [],
            "files": file_list,
            "branches": branches,
            "version_tags": tags,
        }

        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Fetch a model card and metadata from a Hugging Face Hub repo. "
        "Returns the README/model card content plus any structured metadata. "
        "repo_id: HF repo (defaults to config). revision: branch or tag (default 'main')."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "repo_id": {"type": "string", "description": "HF repo ID (default from config)"},
            "revision": {"type": "string", "description": "Branch or tag (default 'main')"},
        },
        "required": [],
    },
)
def fetch_hf_model_card(repo_id: str = "", revision: str = "main") -> str:
    """Fetch model card content and metadata from HF Hub."""
    try:
        from huggingface_hub import HfApi, ModelCard

        target_repo = repo_id or HF_REPO_ID
        if not target_repo:
            return json.dumps({"error": "No HF_REPO_ID configured"})

        card = ModelCard.load(target_repo, revision=revision, token=HF_TOKEN or None)
        card_data = card.data.to_dict() if card.data else {}

        # Try to load config.json if it exists
        api = HfApi(token=HF_TOKEN or None)
        config = None
        try:
            config_path = api.hf_hub_download(
                repo_id=target_repo,
                filename="config.json",
                revision=revision,
            )
            import json as _json
            with open(config_path) as f:
                config = _json.load(f)
        except Exception:
            pass

        return json.dumps({
            "repo_id": target_repo,
            "revision": revision,
            "card_content": card.text[:5000] if card.text else "",
            "metadata": card_data,
            "config": config,
        }, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Load a specific artifact (JSON file) from a Hugging Face Hub repo. "
        "Useful for reading experiment configs, metrics, or results stored alongside models. "
        "filename: path within the repo (e.g. 'experiment.json', 'metrics.json')."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "File path within the repo"},
            "repo_id": {"type": "string", "description": "HF repo ID (default from config)"},
            "revision": {"type": "string", "description": "Branch or tag (default 'main')"},
        },
        "required": ["filename"],
    },
)
def fetch_hf_artifact(filename: str, repo_id: str = "", revision: str = "main") -> str:
    """Download and return the contents of a JSON file from HF Hub."""
    try:
        from huggingface_hub import HfApi

        target_repo = repo_id or HF_REPO_ID
        if not target_repo:
            return json.dumps({"error": "No HF_REPO_ID configured"})

        api = HfApi(token=HF_TOKEN or None)
        local_path = api.hf_hub_download(
            repo_id=target_repo,
            filename=filename,
            revision=revision,
        )

        with open(local_path) as f:
            content = f.read()

        # Try to parse as JSON, fall back to raw text
        try:
            data = json.loads(content)
            return json.dumps({"filename": filename, "data": data}, indent=2, default=str)
        except json.JSONDecodeError:
            return json.dumps({"filename": filename, "content": content[:5000]})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
