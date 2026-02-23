"""
Clawbot change tracker — sandbox git repo + JSONL audit log.

Tracks all code changes made by clawbots (new components, agents, prompts,
configs) in an isolated git repository separate from the main synth-city and
open-synth-miner repos.  This prevents clawbot-authored code from polluting
mainline history while providing full version control and auditability.

The tracker maintains:
  1. A **mirror git repo** at ``CHANGE_TRACKING_DIR`` with directory structure
     mirroring both repos (``synth-city/`` and ``open-synth-miner/``).
  2. A **JSONL audit log** recording every tracked write with metadata
     (bot_id, timestamp, tool, file path, operation type, commit hash).

Usage::

    from pipeline.change_tracker import get_tracker

    tracker = get_tracker()
    tracker.track_write(
        repo="open-synth-miner",
        rel_path="src/models/components/wavelet_block.py",
        content=code,
        bot_id="alpha",
        tool_name="write_component",
    )
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import CHANGE_TRACKING_DIR, CHANGE_TRACKING_ENABLED

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audit log entry
# ---------------------------------------------------------------------------

@dataclass
class ChangeEntry:
    """A single tracked change."""

    timestamp: float
    bot_id: str
    repo: str  # "synth-city" or "open-synth-miner"
    rel_path: str
    operation: str  # "write", "overwrite", "delete"
    tool_name: str
    bytes_written: int
    commit_hash: str  # empty if git commit failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "iso_time": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "bot_id": self.bot_id,
            "repo": self.repo,
            "rel_path": self.rel_path,
            "operation": self.operation,
            "tool_name": self.tool_name,
            "bytes_written": self.bytes_written,
            "commit_hash": self.commit_hash,
        }


# ---------------------------------------------------------------------------
# ChangeTracker
# ---------------------------------------------------------------------------

class ChangeTracker:
    """Sandbox git repo that mirrors clawbot-authored code changes."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or CHANGE_TRACKING_DIR).resolve()
        self._audit_log = self._root / "audit.jsonl"
        self._lock = threading.Lock()
        self._initialised = False

    # -- lazy init ----------------------------------------------------------

    def _ensure_init(self) -> None:
        """Create the tracking directory and init git repo on first use."""
        if self._initialised:
            return
        with self._lock:
            if self._initialised:
                return
            self._root.mkdir(parents=True, exist_ok=True)
            git_dir = self._root / ".git"
            if not git_dir.exists():
                self._git("init")
                self._git("config", "user.name", "clawbot-tracker")
                self._git("config", "user.email", "clawbot@synth-city.local")
                # Initial empty commit so we always have a HEAD
                self._git("commit", "--allow-empty", "-m", "init: change tracking repo")
                logger.info("Initialised change tracking repo at %s", self._root)
            self._initialised = True

    # -- git helpers --------------------------------------------------------

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        """Run a git command in the tracking repo."""
        cmd = ["git", "-C", str(self._root), *args]
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
            timeout=30,
        )

    # -- core tracking ------------------------------------------------------

    def track_write(
        self,
        repo: str,
        rel_path: str,
        content: str,
        bot_id: str = "unknown",
        tool_name: str = "unknown",
    ) -> ChangeEntry:
        """Mirror a file write into the tracking repo and commit it.

        Parameters
        ----------
        repo:
            Which source repo this file belongs to (``"synth-city"`` or
            ``"open-synth-miner"``).
        rel_path:
            Path relative to the source repo root (e.g.
            ``"src/models/components/wavelet_block.py"``).
        content:
            Full file content.
        bot_id:
            Identifier of the bot that authored the change.
        tool_name:
            Name of the tool that triggered the write.

        Returns
        -------
        ChangeEntry with metadata including the git commit hash.
        """
        if not CHANGE_TRACKING_ENABLED:
            return ChangeEntry(
                timestamp=time.time(),
                bot_id=bot_id,
                repo=repo,
                rel_path=rel_path,
                operation="write",
                tool_name=tool_name,
                bytes_written=len(content),
                commit_hash="",
            )

        self._ensure_init()

        mirror_path = self._root / repo / rel_path
        operation = "overwrite" if mirror_path.exists() else "write"
        ts = time.time()

        with self._lock:
            # Write the file into the mirror repo
            mirror_path.parent.mkdir(parents=True, exist_ok=True)
            mirror_path.write_text(content, encoding="utf-8")

            # Stage and commit
            commit_hash = self._commit_change(
                repo=repo,
                rel_path=rel_path,
                bot_id=bot_id,
                tool_name=tool_name,
                operation=operation,
            )

        entry = ChangeEntry(
            timestamp=ts,
            bot_id=bot_id,
            repo=repo,
            rel_path=rel_path,
            operation=operation,
            tool_name=tool_name,
            bytes_written=len(content),
            commit_hash=commit_hash,
        )
        self._append_audit(entry)
        return entry

    def track_delete(
        self,
        repo: str,
        rel_path: str,
        bot_id: str = "unknown",
        tool_name: str = "unknown",
    ) -> ChangeEntry:
        """Record a file deletion in the tracking repo."""
        if not CHANGE_TRACKING_ENABLED:
            return ChangeEntry(
                timestamp=time.time(),
                bot_id=bot_id,
                repo=repo,
                rel_path=rel_path,
                operation="delete",
                tool_name=tool_name,
                bytes_written=0,
                commit_hash="",
            )

        self._ensure_init()
        ts = time.time()
        mirror_path = self._root / repo / rel_path

        with self._lock:
            if mirror_path.exists():
                mirror_path.unlink()
            commit_hash = self._commit_change(
                repo=repo,
                rel_path=rel_path,
                bot_id=bot_id,
                tool_name=tool_name,
                operation="delete",
            )

        entry = ChangeEntry(
            timestamp=ts,
            bot_id=bot_id,
            repo=repo,
            rel_path=rel_path,
            operation="delete",
            tool_name=tool_name,
            bytes_written=0,
            commit_hash=commit_hash,
        )
        self._append_audit(entry)
        return entry

    def _commit_change(
        self,
        repo: str,
        rel_path: str,
        bot_id: str,
        tool_name: str,
        operation: str,
    ) -> str:
        """Stage all changes and commit.  Returns the commit hash or ``""``."""
        try:
            self._git("add", "-A")
            msg = (
                f"{operation}: {repo}/{rel_path}\n\n"
                f"bot: {bot_id}\n"
                f"tool: {tool_name}\n"
                f"time: {datetime.now(timezone.utc).isoformat()}"
            )
            result = self._git("commit", "-m", msg, check=False)
            if result.returncode != 0:
                # Nothing to commit (e.g. identical content)
                if "nothing to commit" in result.stdout:
                    return ""
                logger.warning(
                    "git commit failed (rc=%d): %s", result.returncode, result.stderr[:500],
                )
                return ""
            # Extract commit hash
            hash_result = self._git("rev-parse", "HEAD", check=False)
            return hash_result.stdout.strip() if hash_result.returncode == 0 else ""
        except Exception as exc:
            logger.warning("Failed to commit tracked change: %s", exc)
            return ""

    # -- audit log ----------------------------------------------------------

    def _append_audit(self, entry: ChangeEntry) -> None:
        """Append an entry to the JSONL audit log."""
        try:
            with open(self._audit_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as exc:
            logger.warning("Failed to write audit log: %s", exc)

    # -- query interface ----------------------------------------------------

    def get_audit_log(
        self,
        limit: int = 50,
        bot_id: str | None = None,
        repo: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read the most recent audit log entries with optional filtering."""
        if not self._audit_log.exists():
            return []
        try:
            lines = self._audit_log.read_text(encoding="utf-8").strip().splitlines()
            entries = [json.loads(line) for line in lines if line.strip()]
            if bot_id:
                entries = [e for e in entries if e.get("bot_id") == bot_id]
            if repo:
                entries = [e for e in entries if e.get("repo") == repo]
            return entries[-limit:]
        except Exception as exc:
            logger.warning("Failed to read audit log: %s", exc)
            return []

    def get_change_diff(self, commit_hash: str) -> dict[str, Any]:
        """Return the diff for a specific commit."""
        self._ensure_init()
        try:
            result = self._git("show", "--stat", "--patch", commit_hash, check=False)
            if result.returncode != 0:
                return {"error": f"Failed to get diff: {result.stderr[:500]}"}
            output = result.stdout
            if len(output) > 50_000:
                output = output[:50_000] + "\n\n... [TRUNCATED]"
            return {"commit": commit_hash, "diff": output}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def get_file_history(
        self,
        repo: str,
        rel_path: str,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        """Return git log entries for a specific file in the tracking repo."""
        self._ensure_init()
        mirror_rel = f"{repo}/{rel_path}"
        try:
            result = self._git(
                "log",
                f"-{limit}",
                "--format=%H|%ai|%s",
                "--",
                mirror_rel,
                check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return []
            commits = []
            for line in result.stdout.strip().splitlines():
                parts = line.split("|", 2)
                if len(parts) == 3:
                    commits.append({
                        "commit": parts[0],
                        "date": parts[1],
                        "message": parts[2],
                    })
            return commits
        except Exception as exc:
            logger.warning("Failed to get file history: %s", exc)
            return []

    def get_log(self, limit: int = 50) -> list[dict[str, str]]:
        """Return recent git log entries from the tracking repo."""
        self._ensure_init()
        try:
            result = self._git(
                "log",
                f"-{limit}",
                "--format=%H|%ai|%s",
                check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return []
            commits = []
            for line in result.stdout.strip().splitlines():
                parts = line.split("|", 2)
                if len(parts) == 3:
                    commits.append({
                        "commit": parts[0],
                        "date": parts[1],
                        "message": parts[2],
                    })
            return commits
        except Exception as exc:
            logger.warning("Failed to get log: %s", exc)
            return []

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics about tracked changes."""
        entries = self.get_audit_log(limit=0)  # 0 = all
        if not entries:
            return {
                "total_changes": 0,
                "bots": [],
                "repos": {},
                "tracking_dir": str(self._root),
            }

        bot_ids = sorted(set(e.get("bot_id", "unknown") for e in entries))
        repo_counts: dict[str, int] = {}
        for e in entries:
            r = e.get("repo", "unknown")
            repo_counts[r] = repo_counts.get(r, 0) + 1

        return {
            "total_changes": len(entries),
            "bots": bot_ids,
            "repos": repo_counts,
            "oldest": entries[0].get("iso_time", ""),
            "newest": entries[-1].get("iso_time", ""),
            "tracking_dir": str(self._root),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_tracker: ChangeTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker() -> ChangeTracker:
    """Return the global ChangeTracker singleton."""
    global _tracker
    if _tracker is not None:
        return _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = ChangeTracker()
    return _tracker
