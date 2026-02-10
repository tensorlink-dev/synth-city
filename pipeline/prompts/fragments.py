"""
Prompt fragment system — composable markdown sections assembled per agent per task.

Fragments are keyed by (agent_name, channel) and assembled into the final
system prompt.  This keeps prompt engineering centralised and easy to iterate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Fragment:
    """A named, ordered piece of a system prompt."""

    key: str
    content: str
    priority: int = 50  # Lower = earlier in the assembled prompt


# Global fragment store: (agent_name, channel) -> list[Fragment]
_FRAGMENTS: dict[tuple[str, str], list[Fragment]] = {}


def register_fragment(
    agent_name: str,
    channel: str,
    key: str,
    content: str,
    priority: int = 50,
) -> None:
    """Register a prompt fragment for an agent+channel combination."""
    bucket = (agent_name, channel)
    if bucket not in _FRAGMENTS:
        _FRAGMENTS[bucket] = []
    _FRAGMENTS[bucket].append(Fragment(key=key, content=content, priority=priority))


def get_fragments(agent_name: str, channel: str) -> list[Fragment]:
    """Return fragments for the given agent+channel, sorted by priority."""
    frags = _FRAGMENTS.get((agent_name, channel), [])
    # Also include wildcard channel fragments
    frags = frags + _FRAGMENTS.get((agent_name, "*"), [])
    return sorted(frags, key=lambda f: f.priority)


def assemble_prompt(agent_name: str, channel: str, task: dict[str, Any] | None = None) -> str:
    """Assemble a full system prompt from registered fragments.

    Supports ``{variable}`` placeholders in fragment content, filled from *task*.
    """
    frags = get_fragments(agent_name, channel)
    parts = []
    for frag in frags:
        content = frag.content
        if task:
            try:
                content = content.format_map(TaskFormatter(task))
            except (KeyError, ValueError):
                pass  # Leave unresolved placeholders as-is
        parts.append(content)
    return "\n\n".join(parts)


class TaskFormatter(dict):
    """Dict subclass that returns the key itself for missing format keys."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
