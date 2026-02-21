"""
Agent design tools — create, read, and list pipeline agents and their prompts.

These tools give the AgentDesigner (or any agent with access) the ability to
create brand-new pipeline agents by writing agent classes and prompt modules
that follow the BaseAgentWrapper pattern.
"""

from __future__ import annotations

import ast
import json
import logging
import threading
from pathlib import Path

from pipeline.tools.registry import tool

logger = logging.getLogger(__name__)

_AGENTS_DIR = Path("pipeline/agents")
_PROMPTS_DIR = Path("pipeline/prompts")

# Reentrant lock for shared writes — agent/prompt files are shared across bots
_agent_write_lock = threading.RLock()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str, suffix: str = ".py") -> str:
    """Ensure *name* ends with the expected suffix."""
    if not name.endswith(suffix):
        name += suffix
    return name


def _validate_python(code: str) -> str | None:
    """Return an error message if *code* is not valid Python, else ``None``."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError at line {exc.lineno}: {exc.msg}"
    return None


# ---------------------------------------------------------------------------
# Agent files
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Write a new pipeline agent class into pipeline/agents/. "
        "The file must define a subclass of BaseAgentWrapper with agent_name, "
        "build_system_prompt(), and build_tools(). "
        "filename: e.g. 'my_agent.py'. "
        "code: the full Python source code."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": (
                    "Filename for the agent module (e.g. 'evaluator.py'). "
                    "Written to pipeline/agents/."
                ),
            },
            "code": {
                "type": "string",
                "description": "Full Python source code for the agent class.",
            },
        },
        "required": ["filename", "code"],
    },
)
def write_agent(filename: str, code: str) -> str:
    """Write an agent module into pipeline/agents/."""
    try:
        filename = _safe_filename(filename)
        err = _validate_python(code)
        if err:
            return json.dumps({"error": f"Invalid Python: {err}"})

        target = _AGENTS_DIR / filename
        with _agent_write_lock:
            _ensure_dir(target.parent)
            target.write_text(code, encoding="utf-8")

        return json.dumps({
            "status": "written",
            "path": str(target),
            "bytes": len(code),
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Read an existing agent module from pipeline/agents/ to study its pattern. "
        "filename: e.g. 'planner.py' or 'debugger.py'."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Filename in pipeline/agents/ (e.g. 'planner.py').",
            },
        },
        "required": ["filename"],
    },
)
def read_agent(filename: str) -> str:
    """Read an agent source file for reference."""
    try:
        filename = _safe_filename(filename)
        target = _AGENTS_DIR / filename
        if not target.exists():
            return json.dumps({"error": f"File not found: {target}"})
        content = target.read_text(encoding="utf-8")
        if len(content) > 30_000:
            content = content[:30_000] + f"\n\n... [TRUNCATED — {len(content)} total chars]"
        return content
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description="List all agent modules in pipeline/agents/.",
)
def list_agents() -> str:
    """List Python files in pipeline/agents/."""
    try:
        if not _AGENTS_DIR.exists():
            return json.dumps({"error": f"Directory not found: {_AGENTS_DIR}"})
        files = sorted(
            f.name for f in _AGENTS_DIR.iterdir()
            if f.is_file() and f.suffix == ".py" and f.name != "__init__.py"
        )
        return json.dumps({"agents_dir": str(_AGENTS_DIR), "files": files})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------------
# Prompt files
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Write a prompt module into pipeline/prompts/ for a new agent. "
        "The file should call register_fragment() to register prompt fragments. "
        "filename: e.g. 'my_agent_prompts.py'. "
        "code: the full Python source code."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": (
                    "Filename for the prompt module (e.g. 'evaluator_prompts.py'). "
                    "Written to pipeline/prompts/."
                ),
            },
            "code": {
                "type": "string",
                "description": "Full Python source code for the prompt module.",
            },
        },
        "required": ["filename", "code"],
    },
)
def write_agent_prompt(filename: str, code: str) -> str:
    """Write a prompt module into pipeline/prompts/."""
    try:
        filename = _safe_filename(filename)
        err = _validate_python(code)
        if err:
            return json.dumps({"error": f"Invalid Python: {err}"})

        target = _PROMPTS_DIR / filename
        with _agent_write_lock:
            _ensure_dir(target.parent)
            target.write_text(code, encoding="utf-8")

        return json.dumps({
            "status": "written",
            "path": str(target),
            "bytes": len(code),
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Read an existing prompt module from pipeline/prompts/ to study its pattern. "
        "filename: e.g. 'planner_prompts.py'."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Filename in pipeline/prompts/ (e.g. 'planner_prompts.py').",
            },
        },
        "required": ["filename"],
    },
)
def read_agent_prompt(filename: str) -> str:
    """Read a prompt module source file for reference."""
    try:
        filename = _safe_filename(filename)
        target = _PROMPTS_DIR / filename
        if not target.exists():
            return json.dumps({"error": f"File not found: {target}"})
        content = target.read_text(encoding="utf-8")
        if len(content) > 30_000:
            content = content[:30_000] + f"\n\n... [TRUNCATED — {len(content)} total chars]"
        return content
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description="List all prompt modules in pipeline/prompts/.",
)
def list_agent_prompts() -> str:
    """List Python files in pipeline/prompts/."""
    try:
        if not _PROMPTS_DIR.exists():
            return json.dumps({"error": f"Directory not found: {_PROMPTS_DIR}"})
        files = sorted(
            f.name for f in _PROMPTS_DIR.iterdir()
            if f.is_file() and f.suffix == ".py" and f.name != "__init__.py"
        )
        return json.dumps({"prompts_dir": str(_PROMPTS_DIR), "files": files})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------------
# Tool registry introspection
# ---------------------------------------------------------------------------

@tool(
    description=(
        "List all registered tool names in the synth-city tool registry. "
        "Use this to see which tools are available when building a new agent's "
        "build_tools() method."
    ),
)
def list_available_tools() -> str:
    """List all tool names currently registered."""
    try:
        from pipeline.tools.registry import all_tool_names
        names = all_tool_names()
        return json.dumps({"tool_count": len(names), "tools": sorted(names)})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
