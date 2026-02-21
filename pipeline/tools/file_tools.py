"""
File I/O tools — the agents produce code exclusively through these tools,
never via raw text in their responses.

Multi-bot support:
    When running via the bridge with a bot session active, file paths are
    resolved relative to the per-bot workspace (``workspace/{bot_id}/``).
    Write operations use per-file locks to prevent concurrent corruption.
"""

from __future__ import annotations

import threading
from pathlib import Path

from config import WORKSPACE_DIR
from pipeline.tools.registry import tool

# Per-file write locks to prevent concurrent writes to the same file
_file_locks: dict[str, threading.Lock] = {}
_file_locks_guard = threading.Lock()


def _get_file_lock(path: Path) -> threading.Lock:
    """Return a lock for the given file path (creating one if needed)."""
    key = str(path)
    with _file_locks_guard:
        lock = _file_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _file_locks[key] = lock
        return lock


def _workspace_root() -> Path:
    """Return the workspace root for the current bot, or the global default."""
    from integrations.openclaw.bot_sessions import get_current_session

    bot = get_current_session()
    if bot is not None:
        return bot.workspace_dir
    return WORKSPACE_DIR


def _resolve(path: str) -> Path:
    """Resolve *path* relative to the workspace, preventing escapes."""
    ws = _workspace_root()
    resolved = (ws / path).resolve()
    if not str(resolved).startswith(str(ws.resolve())):
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


@tool(description="Write content to a file at the given path (relative to workspace).")
def write_file(path: str, content: str) -> str:
    target = _resolve(path)
    lock = _get_file_lock(target)
    with lock:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"


@tool(description="Read the content of a file at the given path (relative to workspace).")
def read_file(path: str) -> str:
    target = _resolve(path)
    if not target.exists():
        return f"ERROR: File not found: {path}"
    content = target.read_text(encoding="utf-8")
    # Ephemeral compression: truncate very large files
    max_chars = 30_000
    if len(content) > max_chars:
        return content[:max_chars] + f"\n\n... [TRUNCATED — {len(content)} total chars]"
    return content


@tool(description="List files in a directory (relative to workspace).")
def list_files(path: str = ".") -> str:
    target = _resolve(path)
    ws = _workspace_root()
    if not target.exists():
        return f"ERROR: Directory not found: {path}"
    entries = []
    for item in sorted(target.iterdir()):
        prefix = "d " if item.is_dir() else "f "
        entries.append(prefix + str(item.relative_to(ws)))
    return "\n".join(entries) if entries else "(empty directory)"


@tool(description="Delete a file at the given path (relative to workspace).")
def delete_file(path: str) -> str:
    target = _resolve(path)
    if not target.exists():
        return f"File not found: {path}"
    target.unlink()
    return f"Deleted {path}"


@tool(description="Append content to a file at the given path (relative to workspace).")
def append_file(path: str, content: str) -> str:
    target = _resolve(path)
    lock = _get_file_lock(target)
    with lock:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(content)
    return f"Appended {len(content)} bytes to {path}"
