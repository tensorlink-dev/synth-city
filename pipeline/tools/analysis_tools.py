"""
Analysis tools — read back from Hippius and HF Hub for historical experiment analysis.

These tools let agents query past results from persistent storage:
  - Hippius: fetch runs, compare CRPS trends, find best historical configs
  - HF Hub: list published models, read model cards, compare versions
  - Experiment scanner: lessons-learned summaries + deduplication fingerprints
"""

from __future__ import annotations

import hashlib
import json
import logging
import traceback
from collections import Counter, defaultdict
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
        from pipeline.tools import hippius_store as _hs

        if _hs._endpoint_unreachable:
            return json.dumps({
                "error": "Hippius endpoint unreachable",
                "error_type": "transient",
                "recoverable": False,
                "total": 0,
                "runs": [],
            })

        keys = _hs._list_keys("experiments/", max_keys=2000)
        if not keys:
            return json.dumps({"total": 0, "order": order, "runs": []}, indent=2)

        experiments: list[dict[str, Any]] = []
        consecutive_failures = 0
        max_consecutive_failures = 3
        for key in keys:
            if _hs._endpoint_unreachable:
                break
            exp = _hs._get_json(key)
            if exp is None:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(
                        "fetch_experiment_runs: %d consecutive download failures, "
                        "stopping early (%d/%d keys fetched)",
                        consecutive_failures, len(experiments), len(keys),
                    )
                    break
                continue
            consecutive_failures = 0
            if isinstance(exp, dict):
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
        from pipeline.tools import hippius_store as _hs

        if _hs._endpoint_unreachable:
            return json.dumps({
                "error": "Hippius endpoint unreachable",
                "error_type": "transient",
                "recoverable": False,
            })

        if run_id == "latest":
            latest = _hs._get_json("pipeline_runs/latest.json")
            if not latest:
                return json.dumps({"error": "No runs found in Hippius"})
            run_id = latest["run_id"]

        summary = _hs._get_json(f"pipeline_runs/{run_id}/summary.json")

        data: dict[str, Any] = {"run_id": run_id}
        if summary:
            data["name"] = summary.get("name", run_id)
            data["created_at"] = summary.get("timestamp", "")
            data["config"] = summary.get("experiment", summary.get("config", {}))
            data["metrics"] = summary.get("metrics", {})

        # Load individual eval results for this run
        exp_keys = _hs._list_keys(f"experiments/{run_id}/")
        experiments = []
        consecutive_failures = 0
        for key in exp_keys:
            if _hs._endpoint_unreachable:
                break
            exp = _hs._get_json(key)
            if exp is None:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    break
                continue
            consecutive_failures = 0
            if isinstance(exp, dict):
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
        from pipeline.tools import hippius_store as _hs

        if _hs._endpoint_unreachable:
            return json.dumps({
                "error": "Hippius endpoint unreachable",
                "error_type": "transient",
                "recoverable": False,
            })

        keys = _hs._list_keys("experiments/", max_keys=2000)

        entries: list[dict[str, Any]] = []
        consecutive_failures = 0
        for key in keys:
            if _hs._endpoint_unreachable:
                break
            exp = _hs._get_json(key)
            if not exp or not isinstance(exp, dict):
                if exp is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        logger.warning(
                            "analyze_experiment_trends: stopping early after %d "
                            "consecutive failures (%d/%d keys fetched)",
                            consecutive_failures, len(entries), len(keys),
                        )
                        break
                continue
            consecutive_failures = 0
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
            "downloads": getattr(info, "downloads", None),
            "likes": getattr(info, "likes", None),
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

        card = ModelCard.load(target_repo, token=HF_TOKEN or None)  # type: ignore[call-arg]
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


# ---------------------------------------------------------------------------
# Experiment history scanner — lessons learned + deduplication
# ---------------------------------------------------------------------------

# All known blocks and heads (mirrored from research_tools fallback data).
_ALL_BLOCKS = [
    "RevIN", "LayerNormBlock", "DLinearBlock", "RNNBlock", "ResConvBlock",
    "BiTCNBlock", "SDEEvolutionBlock", "GRUBlock", "LSTMBlock", "FourierBlock",
    "TransformerBlock", "TimeMixerBlock", "Unet1DBlock", "TransformerEncoder",
    "TimesNetBlock",
]
_ALL_HEADS = [
    "GBMHead", "SDEHead", "SimpleHorizonHead", "HorizonHead",
    "NeuralBridgeHead", "NeuralSDEHead",
]


def _fingerprint(experiment: dict) -> str:
    """Compute a stable fingerprint for an experiment config.

    Captures architecture (blocks + head + d_model) and key hyperparameters
    (lr, horizon, seq_len) so that near-identical experiments can be detected.
    """
    model = experiment.get("model", {})
    backbone = model.get("backbone", {})
    training = experiment.get("training", {})

    blocks = backbone.get("blocks", [])
    if not isinstance(blocks, list):
        blocks = []

    head_cfg = model.get("head", {})
    head_name = head_cfg.get("_target_", "") if isinstance(head_cfg, dict) else str(head_cfg)
    if "." in head_name:
        head_name = head_name.rsplit(".", 1)[-1]

    parts = [
        "|".join(blocks),
        head_name,
        str(backbone.get("d_model", "")),
        str(training.get("lr", "")),
        str(training.get("horizon", "")),
        str(backbone.get("seq_len", "")),
    ]
    raw = "::".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _extract_config_features(experiment: dict) -> dict[str, Any]:
    """Extract the architecturally relevant fields from an experiment config."""
    model = experiment.get("model", {})
    backbone = model.get("backbone", {})
    training = experiment.get("training", {})
    head_cfg = model.get("head", {})
    head_name = head_cfg.get("_target_", "unknown") if isinstance(head_cfg, dict) else str(head_cfg)
    if "." in head_name:
        head_name = head_name.rsplit(".", 1)[-1]
    return {
        "blocks": backbone.get("blocks", []),
        "head": head_name,
        "d_model": backbone.get("d_model"),
        "lr": training.get("lr"),
        "horizon": training.get("horizon"),
        "seq_len": backbone.get("seq_len"),
    }


def _build_scan_result(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyse a list of loaded experiment records and produce a structured summary.

    Each record is expected to have: name, experiment (config dict), result (dict
    with metrics), timestamp, run_id.
    """
    if not experiments:
        return {
            "total_experiments": 0,
            "successful": 0,
            "failed": 0,
            "lessons": "No prior experiments found. Start fresh.",
            "top_configs": [],
            "block_stats": {},
            "head_stats": {},
            "failure_patterns": [],
            "untried_blocks": list(_ALL_BLOCKS),
            "untried_heads": list(_ALL_HEADS),
            "untried_block_head_pairs": [],
            "duplicates": {},
            "duplicate_count": 0,
        }

    # ------------------------------------------------------------------
    # 1. Parse all experiments into a normalised form
    # ------------------------------------------------------------------
    parsed: list[dict[str, Any]] = []
    fingerprints: dict[str, list[str]] = defaultdict(list)  # fp → [names]

    for rec in experiments:
        config = rec.get("experiment", rec.get("config", {}))
        if not isinstance(config, dict):
            continue
        result = rec.get("result", {})
        if not isinstance(result, dict):
            result = {}
        metrics = result.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        crps = metrics.get("crps")
        status = result.get("status", "unknown")
        error = result.get("error", "")
        features = _extract_config_features(config)
        fp = _fingerprint(config)
        name = rec.get("name", "unknown")

        fingerprints[fp].append(name)
        parsed.append({
            "name": name,
            "run_id": rec.get("run_id", "unknown"),
            "timestamp": rec.get("timestamp", ""),
            "crps": crps,
            "status": status,
            "error": str(error)[:200] if error else "",
            "fingerprint": fp,
            **features,
        })

    # ------------------------------------------------------------------
    # 2. Top configs (best CRPS)
    # ------------------------------------------------------------------
    successful = [p for p in parsed if p["crps"] is not None and p["status"] != "error"]
    successful.sort(key=lambda e: e["crps"])
    top_configs = successful[:5]

    # ------------------------------------------------------------------
    # 3. Failure patterns
    # ------------------------------------------------------------------
    failures = [p for p in parsed if p["status"] == "error" or p.get("error")]
    error_counter: Counter[str] = Counter()
    for f in failures:
        err_text = f["error"]
        err_lower = err_text.lower()
        # Categorise the error
        if "d_model" in err_text and "divisible" in err_text:
            error_counter["d_model not divisible by nhead"] += 1
        elif "oom" in err_lower or "out of memory" in err_lower:
            error_counter["out of memory"] += 1
        elif "nan" in err_lower:
            error_counter["NaN loss / degenerate output"] += 1
        elif "Unknown block" in err_text or "Unknown head" in err_text:
            error_counter["unknown component name"] += 1
        elif err_text:
            error_counter[err_text[:80]] += 1
        else:
            error_counter["unspecified error"] += 1

    failure_patterns = [
        {"error": err, "count": cnt}
        for err, cnt in error_counter.most_common(10)
    ]

    # ------------------------------------------------------------------
    # 4. Block & head performance stats
    # ------------------------------------------------------------------
    block_crps: dict[str, list[float]] = defaultdict(list)
    head_crps: dict[str, list[float]] = defaultdict(list)

    for p in successful:
        for block in p.get("blocks", []):
            block_crps[block].append(p["crps"])
        head_crps[p.get("head", "unknown")].append(p["crps"])

    block_stats = {
        b: {"mean_crps": round(sum(v) / len(v), 6), "best_crps": round(min(v), 6), "count": len(v)}
        for b, v in sorted(block_crps.items(), key=lambda kv: min(kv[1]))
    }
    head_stats = {
        h: {"mean_crps": round(sum(v) / len(v), 6), "best_crps": round(min(v), 6), "count": len(v)}
        for h, v in sorted(head_crps.items(), key=lambda kv: min(kv[1]))
    }

    # ------------------------------------------------------------------
    # 5. Untried combinations
    # ------------------------------------------------------------------
    tried_blocks = set()
    tried_heads = set()
    tried_pairs: set[tuple[str, str]] = set()
    for p in parsed:
        for block in p.get("blocks", []):
            tried_blocks.add(block)
            tried_pairs.add((block, p.get("head", "unknown")))
        tried_heads.add(p.get("head", "unknown"))

    untried_blocks = [b for b in _ALL_BLOCKS if b not in tried_blocks]
    untried_heads = [h for h in _ALL_HEADS if h not in tried_heads]

    # Find block+head pairs never tried (only for blocks/heads that have each
    # been tried individually, to avoid a combinatorial explosion).
    untried_pairs = []
    for block in sorted(tried_blocks):
        for head in sorted(tried_heads):
            if (block, head) not in tried_pairs:
                untried_pairs.append({"block": block, "head": head})

    # ------------------------------------------------------------------
    # 6. Duplicate detection
    # ------------------------------------------------------------------
    duplicates = {
        fp: names for fp, names in fingerprints.items() if len(names) > 1
    }

    # ------------------------------------------------------------------
    # 7. Synthesise lessons
    # ------------------------------------------------------------------
    lessons: list[str] = []

    if top_configs:
        best = top_configs[0]
        lessons.append(
            f"Best CRPS so far: {best['crps']} using "
            f"{'+'.join(best.get('blocks', []))} → {best.get('head', '?')} "
            f"(d_model={best.get('d_model')}, lr={best.get('lr')})."
        )

    if head_stats:
        best_head = min(head_stats.items(), key=lambda kv: kv[1]["best_crps"])
        worst_head = max(head_stats.items(), key=lambda kv: kv[1]["mean_crps"])
        lessons.append(
            f"Best head: {best_head[0]} (best CRPS {best_head[1]['best_crps']}). "
            f"Weakest head: {worst_head[0]} (mean CRPS {worst_head[1]['mean_crps']})."
        )

    if block_stats:
        best_block = min(block_stats.items(), key=lambda kv: kv[1]["best_crps"])
        lessons.append(
            f"Best-performing block: {best_block[0]} "
            f"(best CRPS {best_block[1]['best_crps']}, used in {best_block[1]['count']} exps)."
        )

    if failure_patterns:
        top_err = failure_patterns[0]
        lessons.append(
            f"Most common failure: \"{top_err['error']}\" ({top_err['count']}x). "
            f"Avoid configs that trigger this."
        )

    if untried_blocks:
        lessons.append(f"Untried blocks: {', '.join(untried_blocks)}. Consider exploring these.")

    if untried_heads:
        lessons.append(f"Untried heads: {', '.join(untried_heads)}. Consider exploring these.")

    if duplicates:
        lessons.append(
            f"{len(duplicates)} experiment config(s) were run more than once "
            f"(same architecture + hyperparams). Avoid re-running these."
        )

    if len(successful) >= 3:
        d_model_crps: dict[int, list[float]] = defaultdict(list)
        for p in successful:
            if p.get("d_model"):
                d_model_crps[p["d_model"]].append(p["crps"])
        if d_model_crps:
            best_dm = min(d_model_crps.items(), key=lambda kv: min(kv[1]))
            lessons.append(
                f"Best d_model so far: {best_dm[0]} (best CRPS {min(best_dm[1]):.6f} "
                f"across {len(best_dm[1])} experiments)."
            )

    return {
        "total_experiments": len(parsed),
        "successful": len(successful),
        "failed": len(failures),
        "lessons": "\n".join(f"- {line}" for line in lessons) if lessons else "Insufficient data.",
        "top_configs": [
            {
                "name": c["name"],
                "crps": c["crps"],
                "blocks": c.get("blocks", []),
                "head": c.get("head"),
                "d_model": c.get("d_model"),
                "lr": c.get("lr"),
                "fingerprint": c["fingerprint"],
            }
            for c in top_configs
        ],
        "block_stats": block_stats,
        "head_stats": head_stats,
        "failure_patterns": failure_patterns,
        "untried_blocks": untried_blocks,
        "untried_heads": untried_heads,
        "untried_block_head_pairs": untried_pairs[:20],
        "duplicates": {fp: names for fp, names in list(duplicates.items())[:10]},
        "duplicate_count": len(duplicates),
    }


@tool(
    description=(
        "Scan ALL prior experiments and produce a structured lessons-learned summary. "
        "Returns: best configs, failure patterns, block/head performance stats, "
        "untried combinations, and duplicate detection. "
        "Call this BEFORE planning new experiments to avoid repeating mistakes. "
        "limit: max experiments to scan (default 100)."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max experiments to scan (default 100)",
            },
        },
        "required": [],
    },
)
def scan_experiment_history(limit: int = 100) -> str:
    """Scan all prior experiments and return a lessons-learned summary."""
    try:
        from pipeline.tools import hippius_store as _hs

        if _hs._endpoint_unreachable:
            return json.dumps({
                "error": "Hippius endpoint unreachable",
                "error_type": "transient",
                "recoverable": False,
            })

        keys = _hs._list_keys("experiments/", max_keys=2000)
        if not keys:
            return json.dumps(_build_scan_result([]), indent=2)

        experiments: list[dict[str, Any]] = []
        consecutive_failures = 0
        for key in keys:
            if _hs._endpoint_unreachable:
                break
            exp = _hs._get_json(key)
            if exp is None:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    logger.warning(
                        "scan_experiment_history: stopping early after %d "
                        "consecutive failures (%d/%d keys fetched)",
                        consecutive_failures, len(experiments), len(keys),
                    )
                    break
                continue
            consecutive_failures = 0
            if isinstance(exp, dict):
                experiments.append(exp)
            if len(experiments) >= limit:
                break

        result = _build_scan_result(experiments)
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        return json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


# ---------------------------------------------------------------------------
# Fingerprint cache — avoids re-downloading all experiments on every
# check_experiment_novelty call.  Populated lazily, refreshed every 5 min.
# ---------------------------------------------------------------------------
_fp_cache: dict[str, list[dict[str, Any]]] = {}  # fingerprint → [{name, run_id, crps, ...}]
_fp_cache_ts: float = 0.0
_FP_CACHE_TTL: float = 300.0  # seconds


def _refresh_fp_cache() -> dict[str, list[dict[str, Any]]]:
    """Rebuild the fingerprint cache from Hippius.

    Downloads all experiment records once, fingerprints each, and stores the
    results in a module-level dict.  Subsequent calls to check_experiment_novelty
    do a simple dict lookup instead of a full S3 scan.
    """
    import time as _time

    global _fp_cache, _fp_cache_ts
    now = _time.monotonic()
    if _fp_cache and (now - _fp_cache_ts) < _FP_CACHE_TTL:
        return _fp_cache

    from pipeline.tools import hippius_store as _hs

    if _hs._endpoint_unreachable:
        return _fp_cache  # return stale cache if any

    keys = _hs._list_keys("experiments/", max_keys=2000)
    new_cache: dict[str, list[dict[str, Any]]] = defaultdict(list)
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

        config = exp.get("experiment", exp.get("config", {}))
        if not isinstance(config, dict):
            continue
        fp = _fingerprint(config)
        result = exp.get("result", {})
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        new_cache[fp].append({
            "name": exp.get("name", "unknown"),
            "run_id": exp.get("run_id", "unknown"),
            "timestamp": exp.get("timestamp", ""),
            "crps": metrics.get("crps") if isinstance(metrics, dict) else None,
            "status": (
                result.get("status", "unknown")
                if isinstance(result, dict) else "unknown"
            ),
        })

    _fp_cache = dict(new_cache)
    _fp_cache_ts = now
    return _fp_cache


@tool(
    description=(
        "Check if an experiment config has already been tried before. "
        "Returns the fingerprint and any matching prior experiments with their CRPS. "
        "Use this before running a new experiment to avoid duplicates. "
        "experiment: the experiment config as a JSON string."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "experiment": {
                "type": "string",
                "description": "Experiment config JSON to check",
            },
        },
        "required": ["experiment"],
    },
)
def check_experiment_novelty(experiment: str) -> str:
    """Check whether an experiment config has been tried before."""
    try:
        exp_dict = json.loads(experiment) if isinstance(experiment, str) else experiment
        target_fp = _fingerprint(exp_dict)
        target_features = _extract_config_features(exp_dict)

        from pipeline.tools import hippius_store as _hs

        if _hs._endpoint_unreachable and not _fp_cache:
            return json.dumps({
                "fingerprint": target_fp,
                "is_novel": True,
                "note": "Cannot verify — Hippius unreachable. Proceeding as novel.",
            })

        cache = _refresh_fp_cache()
        matches = cache.get(target_fp, [])

        is_novel = len(matches) == 0
        response: dict[str, Any] = {
            "fingerprint": target_fp,
            "is_novel": is_novel,
            "architecture": target_features,
        }

        if matches:
            sorted_matches = sorted(
                matches,
                key=lambda m: m["crps"] if m["crps"] is not None else float("inf"),
            )
            response["prior_runs"] = sorted_matches
            response["best_prior_crps"] = sorted_matches[0]["crps"]
            response["times_tried"] = len(sorted_matches)
            response["recommendation"] = (
                "This exact config has been tried before. "
                "Consider changing blocks, head, d_model, or lr to explore new territory."
            )
        else:
            response["recommendation"] = "Novel config — proceed with training."

        return json.dumps(response, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
