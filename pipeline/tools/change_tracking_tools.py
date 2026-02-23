"""
Change tracking tools — query the clawbot change history.

Exposes the ChangeTracker's query interface as agent-callable tools so bots
can inspect what has been written, by whom, and when.
"""

from __future__ import annotations

import json
import logging

from pipeline.tools.registry import tool

logger = logging.getLogger(__name__)


def _get_bot_id() -> str:
    """Return the active bot ID, or ``"unknown"`` in CLI mode."""
    try:
        from integrations.openclaw.bot_sessions import get_current_session
        session = get_current_session()
        return session.bot_id if session else "unknown"
    except Exception:
        return "unknown"


@tool(
    description=(
        "List recent clawbot-authored code changes with optional filtering by "
        "bot_id and repo. Returns audit log entries showing who changed what and when."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max entries to return (default 30).",
            },
            "bot_id": {
                "type": "string",
                "description": "Filter by bot ID (optional).",
            },
            "repo": {
                "type": "string",
                "description": (
                    "Filter by repo: 'synth-city' or 'open-synth-miner' (optional)."
                ),
            },
        },
    },
)
def list_tracked_changes(
    limit: int = 30,
    bot_id: str = "",
    repo: str = "",
) -> str:
    """List recent tracked changes from the audit log."""
    try:
        from pipeline.change_tracker import get_tracker
        tracker = get_tracker()
        entries = tracker.get_audit_log(
            limit=limit,
            bot_id=bot_id or None,
            repo=repo or None,
        )
        return json.dumps({"count": len(entries), "changes": entries})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Get the full diff for a specific tracked change by commit hash. "
        "Use list_tracked_changes first to find commit hashes."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "commit_hash": {
                "type": "string",
                "description": "The git commit hash to show.",
            },
        },
        "required": ["commit_hash"],
    },
)
def get_change_diff(commit_hash: str) -> str:
    """Show the diff for a tracked change."""
    try:
        from pipeline.change_tracker import get_tracker
        tracker = get_tracker()
        result = tracker.get_change_diff(commit_hash)
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Get the version history of a specific file in the change tracker. "
        "Shows all commits that touched this file."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repo name: 'synth-city' or 'open-synth-miner'.",
            },
            "rel_path": {
                "type": "string",
                "description": (
                    "Path relative to repo root "
                    "(e.g. 'src/models/components/wavelet_block.py')."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max commits to return (default 20).",
            },
        },
        "required": ["repo", "rel_path"],
    },
)
def get_file_change_history(
    repo: str,
    rel_path: str,
    limit: int = 20,
) -> str:
    """Get git log for a tracked file."""
    try:
        from pipeline.change_tracker import get_tracker
        tracker = get_tracker()
        commits = tracker.get_file_history(repo, rel_path, limit=limit)
        return json.dumps({"file": f"{repo}/{rel_path}", "commits": commits})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Get the full git log of all tracked changes in the change tracking repo. "
        "Shows commit hashes, dates, and messages."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max commits to return (default 50).",
            },
        },
    },
)
def get_change_log(limit: int = 50) -> str:
    """Get the git log from the change tracking repo."""
    try:
        from pipeline.change_tracker import get_tracker
        tracker = get_tracker()
        commits = tracker.get_log(limit=limit)
        return json.dumps({"count": len(commits), "commits": commits})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Get summary statistics about all clawbot changes: total changes, "
        "active bots, per-repo counts, and time range."
    ),
)
def get_change_stats() -> str:
    """Get summary statistics about tracked changes."""
    try:
        from pipeline.change_tracker import get_tracker
        tracker = get_tracker()
        stats = tracker.get_stats()
        return json.dumps(stats)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
