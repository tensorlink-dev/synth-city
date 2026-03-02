"""
Publish tools — HF Hub model publishing and W&B experiment tracking.

These tools are opt-in side effects used only when the agent decides a model
is good enough to publish. The Planner/Trainer run zero-side-effect experiments
via ResearchSession; only the Publisher agent invokes these.
"""

from __future__ import annotations

import json
import logging
import traceback

from config import HF_REPO_ID, HF_RESULTS_REPO_ID, HF_TOKEN, TRACKIO_PROJECT, WANDB_PROJECT
from pipeline.tools.registry import tool

logger = logging.getLogger(__name__)


@tool(
    description=(
        "Publish a trained model to Hugging Face Hub and log to W&B. "
        "experiment: the experiment config JSON used to build the model. "
        "crps_score: the CRPS achieved. "
        "repo_id: HF Hub repo (default from config). "
        "This is a SIDE-EFFECT operation — only call when you are confident the model is ready."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "experiment": {"type": "string", "description": "Experiment config JSON"},
            "crps_score": {"type": "number", "description": "CRPS score achieved"},
            "repo_id": {"type": "string", "description": "HF Hub repo ID (optional)"},
        },
        "required": ["experiment", "crps_score"],
    },
)
def publish_model(experiment: str, crps_score: float, repo_id: str = "") -> str:
    """Publish model to HF Hub with W&B tracking."""
    try:
        import wandb
        from omegaconf import OmegaConf
        from osa.models.factory import create_model
        from osa.models.registry import discover_components, registry
        from osa.tracking.hub_manager import HubManager

        if HF_TOKEN:
            from huggingface_hub import login
            login(token=HF_TOKEN, add_to_git_credential=False)

        exp_dict = json.loads(experiment) if isinstance(experiment, str) else experiment
        target_repo = repo_id or HF_REPO_ID
        if not target_repo:
            return json.dumps({
                "error": "No HF_REPO_ID configured."
                " Set it in .env or pass repo_id.",
            })

        # Recreate model from config
        discover_components("src/models/components")
        cfg = OmegaConf.create(exp_dict)
        model = create_model(cfg)

        # Get recipe hash
        recipe = exp_dict["model"]["backbone"]["blocks"]
        block_hash = registry.recipe_hash(recipe)
        head_name = exp_dict["model"]["head"]["_target_"].split(".")[-1]

        # Init W&B
        run = wandb.init(project=WANDB_PROJECT, config=exp_dict)

        # Publish
        manager = HubManager(
            run=run,
            backbone_name="HybridBackbone",
            head_name=head_name,
            block_hash=block_hash,
            recipe=recipe,
            architecture_graph=exp_dict["model"]["backbone"],
            resolved_config=exp_dict,
            repo_id=target_repo,
        )
        hf_link = manager.save_and_push(model=model, crps_score=crps_score)
        report = manager.get_shareable_report(crps_score=crps_score, hf_link=hf_link)

        wandb.finish()

        return json.dumps({
            "status": "published",
            "hf_link": hf_link,
            "report": report,
        }, indent=2)

    except Exception as exc:
        return json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


@tool(
    description=(
        "Log experiment metrics to Trackio for local tracking and dashboard. "
        "Also persists to Hippius for long-term decentralised storage. "
        "Useful for tracking intermediate results."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "experiment_name": {"type": "string", "description": "Name for this experiment run"},
            "metrics": {"type": "string", "description": "Metrics as JSON dict"},
            "config": {"type": "string", "description": "Experiment config as JSON dict"},
        },
        "required": ["experiment_name", "metrics"],
    },
)
def log_to_trackio(experiment_name: str, metrics: str, config: str = "") -> str:
    """Log metrics to Trackio for local tracking and persist to Hippius."""
    try:
        import trackio

        metrics_dict = json.loads(metrics) if isinstance(metrics, str) else metrics
        config_dict = json.loads(config) if config else {}

        trackio.init(
            project=TRACKIO_PROJECT,
            name=experiment_name,
            config=config_dict,
        )
        trackio.log(metrics_dict)
        trackio.finish()

        # Also persist to Hippius for cross-session history
        from pipeline.tools.hippius_store import save_experiment_result

        save_experiment_result(
            name=experiment_name,
            experiment=config_dict,
            result={"metrics": metrics_dict},
        )

        return json.dumps({"status": "logged", "project": TRACKIO_PROJECT})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------------
# Public results sharing — HF Hub Dataset
# ---------------------------------------------------------------------------

def _build_results_payload(limit: int = 200) -> list[dict]:
    """Collect experiment results from Hippius into a flat list of records.

    Each record contains the experiment config, metrics, and provenance
    metadata in a schema designed for easy consumption by downstream pipelines.
    """
    from pipeline.tools import hippius_store as _hs

    keys = _hs._list_keys("experiments/", max_keys=2000)
    if not keys:
        return []

    records: list[dict] = []
    consecutive_failures = 0
    for key in keys:
        if _hs._endpoint_unreachable:
            break
        exp = _hs._get_json(key)
        if exp is None:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break
            continue
        consecutive_failures = 0
        if not isinstance(exp, dict):
            continue

        config = exp.get("experiment", {})
        result = exp.get("result", {}) if isinstance(exp.get("result"), dict) else {}
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}

        model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
        backbone = model_cfg.get("backbone", {}) if isinstance(model_cfg, dict) else {}
        head_cfg = model_cfg.get("head", {}) if isinstance(model_cfg, dict) else {}
        training = config.get("training", {}) if isinstance(config, dict) else {}

        head_name = head_cfg.get("_target_", "") if isinstance(head_cfg, dict) else str(head_cfg)
        if "." in head_name:
            head_name = head_name.rsplit(".", 1)[-1]

        records.append({
            "run_id": exp.get("run_id", ""),
            "name": exp.get("name", ""),
            "timestamp": exp.get("timestamp", ""),
            "blocks": backbone.get("blocks", []),
            "head": head_name,
            "d_model": backbone.get("d_model"),
            "seq_len": backbone.get("seq_len"),
            "feature_dim": backbone.get("feature_dim"),
            "horizon": training.get("horizon"),
            "n_paths": training.get("n_paths"),
            "batch_size": training.get("batch_size"),
            "lr": training.get("lr"),
            "crps": metrics.get("crps"),
            "sharpness": metrics.get("sharpness"),
            "log_likelihood": metrics.get("log_likelihood"),
            "param_count": metrics.get("param_count"),
            "status": result.get("status", "unknown"),
            "experiment_config": config,
        })

        if len(records) >= limit:
            break

    # Sort by CRPS (best first), nulls last
    records.sort(
        key=lambda r: r["crps"] if r["crps"] is not None else float("inf")
    )
    return records


@tool(
    description=(
        "Share experiment results publicly by uploading them as a HF Hub Dataset. "
        "Anyone can then consume results via: "
        "datasets.load_dataset('your-org/synth-city-results'). "
        "repo_id: target HF dataset repo (default from HF_RESULTS_REPO_ID config). "
        "limit: max experiments to include (default 200). "
        "This is a SIDE-EFFECT operation — publishes data to a public HF repo."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "repo_id": {
                "type": "string",
                "description": "HF dataset repo ID (e.g. 'myorg/synth-city-results')",
            },
            "limit": {
                "type": "integer",
                "description": "Max experiments to include (default 200)",
            },
        },
        "required": [],
    },
)
def share_results(repo_id: str = "", limit: int = 200) -> str:
    """Upload experiment results to a public HF Hub Dataset repository.

    The dataset is published in JSONL format with a README that documents the
    schema, making it trivial for other pipelines to ingest::

        from datasets import load_dataset
        ds = load_dataset("your-org/synth-city-results")

    Or with plain ``huggingface_hub``::

        from huggingface_hub import hf_hub_download
        path = hf_hub_download("your-org/synth-city-results", "results.jsonl",
                               repo_type="dataset")
    """
    try:
        from datetime import datetime, timezone

        from huggingface_hub import HfApi

        if HF_TOKEN:
            from huggingface_hub import login
            login(token=HF_TOKEN, add_to_git_credential=False)

        target_repo = repo_id or HF_RESULTS_REPO_ID
        if not target_repo:
            return json.dumps({
                "error": "No dataset repo configured. "
                "Set HF_RESULTS_REPO_ID in .env or pass repo_id.",
            })

        # Collect results from Hippius
        records = _build_results_payload(limit=limit)
        if not records:
            return json.dumps({
                "error": "No experiment results found in Hippius storage.",
            })

        # Build JSONL content
        jsonl_lines = [json.dumps(r, default=str) for r in records]
        jsonl_content = "\n".join(jsonl_lines) + "\n"

        # Build a summary JSON for quick stats
        crps_values = [r["crps"] for r in records if r["crps"] is not None]
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_experiments": len(records),
            "successful": sum(1 for r in records if r["status"] == "ok"),
            "best_crps": min(crps_values) if crps_values else None,
            "mean_crps": (
                round(sum(crps_values) / len(crps_values), 6) if crps_values else None
            ),
            "unique_blocks": sorted(
                {b for r in records for b in r.get("blocks", [])}
            ),
            "unique_heads": sorted(
                {r["head"] for r in records if r.get("head")}
            ),
        }
        summary_content = json.dumps(summary, indent=2, default=str)

        # Build README
        readme = _build_dataset_readme(target_repo, summary)

        # Upload to HF Hub
        api = HfApi(token=HF_TOKEN or None)

        # Create repo if it doesn't exist
        api.create_repo(
            repo_id=target_repo,
            repo_type="dataset",
            exist_ok=True,
        )

        # Upload files
        api.upload_file(
            path_or_fileobj=jsonl_content.encode(),
            path_in_repo="results.jsonl",
            repo_id=target_repo,
            repo_type="dataset",
            commit_message=f"Update results ({len(records)} experiments)",
        )
        api.upload_file(
            path_or_fileobj=summary_content.encode(),
            path_in_repo="summary.json",
            repo_id=target_repo,
            repo_type="dataset",
            commit_message=f"Update summary ({len(records)} experiments)",
        )
        api.upload_file(
            path_or_fileobj=readme.encode(),
            path_in_repo="README.md",
            repo_id=target_repo,
            repo_type="dataset",
            commit_message="Update dataset card",
        )

        return json.dumps({
            "status": "shared",
            "repo_id": target_repo,
            "url": f"https://huggingface.co/datasets/{target_repo}",
            "experiments": len(records),
            "best_crps": summary["best_crps"],
            "usage": (
                f'from datasets import load_dataset\n'
                f'ds = load_dataset("{target_repo}")'
            ),
        }, indent=2)

    except Exception as exc:
        return json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


def _build_dataset_readme(repo_id: str, summary: dict) -> str:
    """Generate a HF dataset card README with schema docs and usage examples."""
    best = summary.get("best_crps", "N/A")
    total = summary.get("total_experiments", 0)
    blocks = ", ".join(summary.get("unique_blocks", []))
    heads = ", ".join(summary.get("unique_heads", []))

    return f"""\
---
license: apache-2.0
task_categories:
  - time-series-forecasting
tags:
  - synth-city
  - bittensor
  - sn50
  - crps
  - probabilistic-forecasting
size_categories:
  - n<1K
---

# synth-city experiment results

Public experiment results from the [synth-city](https://github.com/tensorlink-dev/synth-city) \
autonomous research pipeline for Bittensor Subnet 50 (Synth).

## Quick stats

| Metric | Value |
|--------|-------|
| Total experiments | {total} |
| Best CRPS | {best} |
| Blocks explored | {blocks} |
| Heads explored | {heads} |
| Last updated | {summary.get("generated_at", "N/A")} |

## Usage

### Python (datasets library)

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")

# Browse experiments sorted by CRPS
for row in ds["train"]:
    print(f"{{row['name']}}  CRPS={{row['crps']}}  blocks={{row['blocks']}}")
```

### Python (huggingface_hub — no pandas needed)

```python
from huggingface_hub import hf_hub_download
import json

path = hf_hub_download("{repo_id}", "results.jsonl", repo_type="dataset")
with open(path) as f:
    experiments = [json.loads(line) for line in f]

# Best experiment
best = min(experiments, key=lambda e: e["crps"] or float("inf"))
print(best["blocks"], best["head"], best["crps"])
```

### Build on top of these results

```python
from huggingface_hub import hf_hub_download
import json

path = hf_hub_download("{repo_id}", "results.jsonl", repo_type="dataset")
with open(path) as f:
    experiments = [json.loads(line) for line in f]

# Find best block combinations
from collections import defaultdict
block_scores = defaultdict(list)
for exp in experiments:
    if exp["crps"] is not None:
        for block in exp["blocks"]:
            block_scores[block].append(exp["crps"])

for block, scores in sorted(block_scores.items(), key=lambda kv: min(kv[1])):
    print(f"  {{block}}: best={{min(scores):.6f}}  mean={{sum(scores)/len(scores):.6f}}")
```

## Schema

Each row in `results.jsonl` contains:

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Pipeline run identifier |
| `name` | string | Experiment name |
| `timestamp` | string | ISO 8601 timestamp |
| `blocks` | list[str] | Backbone block names (e.g. `["RevIN", "TransformerBlock"]`) |
| `head` | string | Prediction head name (e.g. `"GBMHead"`) |
| `d_model` | int | Hidden dimension |
| `seq_len` | int | Input sequence length |
| `feature_dim` | int | Input feature dimension |
| `horizon` | int | Prediction horizon (steps) |
| `n_paths` | int | Monte Carlo paths |
| `batch_size` | int | Training batch size |
| `lr` | float | Learning rate |
| `crps` | float | CRPS score (lower is better) |
| `sharpness` | float | Sharpness metric |
| `log_likelihood` | float | Log-likelihood |
| `param_count` | int | Model parameter count |
| `status` | string | `"ok"` or `"error"` |
| `experiment_config` | object | Full experiment config for reproducibility |

## About CRPS

CRPS (Continuous Ranked Probability Score) measures how well a probabilistic
forecast matches reality. **Lower is better.** It rewards both accuracy and
well-calibrated uncertainty estimates.

## License

Apache 2.0
"""
