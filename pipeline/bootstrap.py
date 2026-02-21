"""
Bootstrap required directories for the synth-city pipeline.

When the pipeline starts fresh (e.g. empty Hippius bucket, clean checkout),
several directories are assumed to exist by tools and agents.  This module
ensures they are all created up front so nothing fails on first access.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import WORKSPACE_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local filesystem directories that must exist before agents run
# ---------------------------------------------------------------------------
_REQUIRED_DIRS: list[Path] = [
    # Agent workspace (already created in config.py, but be defensive)
    WORKSPACE_DIR,
    # open-synth-miner component directories (register_tools.py, publish_tools.py)
    Path("src/models/components"),
    # Model YAML configs (register_tools.py)
    Path("configs/model"),
    # Agent modules and prompts (agent_tools.py)
    Path("pipeline/agents"),
    Path("pipeline/prompts"),
]

# ---------------------------------------------------------------------------
# S3 "directory" prefixes to seed in Hippius when the bucket is empty
# ---------------------------------------------------------------------------
_HIPPIUS_PREFIXES: list[str] = [
    "experiments/",
    "pipeline_runs/",
]


def bootstrap_dirs() -> list[str]:
    """Create all required local directories.

    Returns a list of directories that were newly created (already-existing
    dirs are silently skipped).
    """
    created: list[str] = []
    for d in _REQUIRED_DIRS:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(str(d))
            logger.info("Created directory: %s", d)
    if created:
        logger.info("Bootstrapped %d directories: %s", len(created), created)
    else:
        logger.debug("All required directories already exist")
    return created


def bootstrap_hippius() -> list[str]:
    """Seed the Hippius bucket with empty prefix markers if the bucket is new.

    S3 doesn't require directories to exist, but some listing operations
    behave better when the prefixes have at least one object.  We write a
    small ``_init.json`` marker under each prefix so that ``list_objects_v2``
    returns results immediately.

    Returns a list of prefixes that were seeded (empty list if the bucket
    already has content or Hippius is not configured).
    """
    try:
        from pipeline.tools.hippius_store import _ensure_bucket, _get_client, _list_keys, _put_json
    except Exception:
        return []

    client = _get_client()
    if client is None:
        logger.debug("Hippius not configured — skipping bucket bootstrap")
        return []

    if not _ensure_bucket():
        logger.warning("Hippius endpoint unreachable — skipping bucket bootstrap")
        return []

    seeded: list[str] = []
    for prefix in _HIPPIUS_PREFIXES:
        existing = _list_keys(prefix, max_keys=1)
        if not existing:
            marker_key = f"{prefix}_init.json"
            ok = _put_json(marker_key, {"_marker": True, "purpose": "bootstrap"})
            if ok:
                seeded.append(prefix)
                logger.info("Seeded Hippius prefix: %s", prefix)

    if seeded:
        logger.info("Bootstrapped %d Hippius prefixes: %s", len(seeded), seeded)
    else:
        logger.debug("Hippius bucket already has content — no seeding needed")
    return seeded


def bootstrap_all() -> dict[str, list[str]]:
    """Run all bootstrap steps.  Safe to call multiple times (idempotent).

    Returns a dict summarising what was created::

        {"dirs": ["workspace", ...], "hippius": ["experiments/", ...]}
    """
    dirs = bootstrap_dirs()
    hippius = bootstrap_hippius()
    return {"dirs": dirs, "hippius": hippius}
