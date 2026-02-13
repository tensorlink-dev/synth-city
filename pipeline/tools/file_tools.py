"""
File I/O tools — the agents produce code exclusively through these tools,
never via raw text in their responses.
"""

from __future__ import annotations

import os
from pathlib import Path

from config import WORKSPACE_DIR
from pipeline.tools.registry import tool


def _resolve(path: str) -> Path:
    """Resolve *path* relative to the workspace, preventing escapes."""
    resolved = (WORKSPACE_DIR / path).resolve()
    if not str(resolved).startswith(str(WORKSPACE_DIR.resolve())):
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


@tool(description="Write content to a file at the given path (relative to workspace).")
def write_file(path: str, content: str) -> str:
    target = _resolve(path)
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
    if not target.exists():
        return f"ERROR: Directory not found: {path}"
    entries = []
    for item in sorted(target.iterdir()):
        prefix = "d " if item.is_dir() else "f "
        entries.append(prefix + str(item.relative_to(WORKSPACE_DIR)))
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
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(content)
    return f"Appended {len(content)} bytes to {path}"
