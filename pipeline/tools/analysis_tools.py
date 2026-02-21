"""
Analysis tools — read back from W&B and HF Hub for historical experiment analysis.

These tools let agents query past results that were previously write-only sinks:
  - W&B: fetch runs, compare CRPS trends, find best historical configs
  - HF Hub: list published models, read model cards, compare versions
"""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any

from config import HF_REPO_ID, WANDB_PROJECT
from pipeline.tools.registry import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# W&B analysis tools
# ---------------------------------------------------------------------------

def _get_wandb_api():
    """Lazy-load the W&B public API client."""
    import wandb
    return wandb.Api()


@tool(
    description=(
        "Fetch recent runs from Weights & Biases for this project. "
        "Returns run names, configs, metrics, and status. "
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
            "filters": {
                "type": "string",
                "description": 'Optional W&B filter JSON, e.g. {"state": "finished"}',
            },
        },
        "required": [],
    },
)
def fetch_wandb_runs(limit: int = 20, order: str = "best", filters: str = "") -> str:
    """Fetch past experiment runs from W&B."""
    try:
        api = _get_wandb_api()
        filter_dict = json.loads(filters) if filters else {"state": "finished"}

        if order == "best":
            sort_key = "+summary_metrics.crps"
        elif order == "worst":
            sort_key = "-summary_metrics.crps"
        else:
            sort_key = "-created_at"

        runs = api.runs(
            path=WANDB_PROJECT,
            filters=filter_dict,
            order=sort_key,
            per_page=limit,
        )

        results = []
        for run in runs:
            summary = dict(run.summary) if run.summary else {}
            # Strip internal W&B keys
            summary = {k: v for k, v in summary.items() if not k.startswith("_")}
            results.append({
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "duration_s": run.summary.get("_runtime"),
                "config": dict(run.config) if run.config else {},
                "metrics": summary,
                "crps": summary.get("crps"),
                "tags": list(run.tags) if run.tags else [],
                "url": run.url,
            })

        return json.dumps(
            {"total": len(results), "order": order, "runs": results},
            indent=2, default=str,
        )
    except Exception as exc:
        return json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


@tool(
    description=(
        "Get detailed information about a specific W&B run including full metric history. "
        "run_id: the W&B run ID (from fetch_wandb_runs)."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "run_id": {"type": "string", "description": "W&B run ID"},
        },
        "required": ["run_id"],
    },
)
def get_wandb_run_detail(run_id: str) -> str:
    """Get full details for a specific W&B run."""
    try:
        api = _get_wandb_api()
        run = api.run(f"{WANDB_PROJECT}/{run_id}")

        summary = dict(run.summary) if run.summary else {}
        summary = {k: v for k, v in summary.items() if not k.startswith("_")}

        # Fetch metric history
        history_df = run.history(samples=500)
        history = history_df.to_dict(orient="records") if not history_df.empty else []

        return json.dumps({
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "created_at": run.created_at,
            "config": dict(run.config) if run.config else {},
            "metrics": summary,
            "history": history,
            "tags": list(run.tags) if run.tags else [],
            "url": run.url,
        }, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Analyse CRPS trends across W&B runs over time. "
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
def analyze_wandb_trends(limit: int = 50) -> str:
    """Analyse CRPS improvement trends across W&B runs."""
    try:
        api = _get_wandb_api()
        runs = api.runs(
            path=WANDB_PROJECT,
            filters={"state": "finished"},
            order="-created_at",
            per_page=limit,
        )

        entries = []
        for run in runs:
            crps = run.summary.get("crps") if run.summary else None
            if crps is not None:
                entries.append({
                    "run_id": run.id,
                    "name": run.name,
                    "created_at": run.created_at,
                    "crps": crps,
                    "sharpness": run.summary.get("sharpness"),
                    "log_likelihood": run.summary.get("log_likelihood"),
                })

        if not entries:
            return json.dumps({"error": "No finished runs with CRPS found in W&B"})

        # Sort chronologically
        entries.sort(key=lambda e: e["created_at"])

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

        api = HfApi()

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
        refs = list_repo_refs(repo_id=target_repo, repo_type="model")
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

        card = ModelCard.load(target_repo, revision=revision)
        card_data = card.data.to_dict() if card.data else {}

        # Try to load config.json if it exists
        api = HfApi()
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

        api = HfApi()
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
