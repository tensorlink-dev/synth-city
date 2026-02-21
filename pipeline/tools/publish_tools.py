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

from config import HF_REPO_ID, WANDB_PROJECT
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
        from src.models.factory import create_model
        from src.models.registry import discover_components, registry
        from src.tracking.hub_manager import HubManager

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
        "Log experiment metrics to W&B without publishing to HF Hub. "
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
def log_to_wandb(experiment_name: str, metrics: str, config: str = "") -> str:
    """Log metrics to W&B for tracking."""
    try:
        import wandb

        metrics_dict = json.loads(metrics) if isinstance(metrics, str) else metrics
        config_dict = json.loads(config) if config else {}

        run = wandb.init(
            project=WANDB_PROJECT,
            name=experiment_name,
            config=config_dict,
        )
        wandb.log(metrics_dict)
        wandb.finish()

        return json.dumps({"status": "logged", "run_url": run.url})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
