"""
Tool authoring tools — create, validate, and reload pipeline tools at runtime.

These tools give agents the ability to extend the system's tool vocabulary
by writing new @tool-decorated functions, validating them, and dynamically
registering them into the global tool registry.
"""

from __future__ import annotations

import ast
import importlib
import json
import logging
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from typing import Any

from pipeline.tools.registry import all_tool_names, get_tool, tool

logger = logging.getLogger(__name__)

_TOOLS_DIR = Path("pipeline/tools")

# Reentrant lock for shared writes
_tool_write_lock = threading.RLock()

# Imports / calls considered dangerous in authored tools
_DANGEROUS_PATTERNS: set[str] = {
    "os.system",
    "subprocess.Popen",
    "eval",
    "exec",
    "__import__",
    "compile",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str, suffix: str = ".py") -> str:
    """Ensure *name* ends with the expected suffix and has no path separators."""
    if not name.endswith(suffix):
        name += suffix
    if "/" in name or "\\" in name:
        raise ValueError(f"Filename must not contain path separators: {name!r}")
    return name


def _validate_python(code: str) -> str | None:
    """Return an error message if *code* is not valid Python, else ``None``."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError at line {exc.lineno}: {exc.msg}"
    return None


def _check_tool_decorator(code: str) -> str | None:
    """Check that at least one function in *code* is decorated with @tool.

    Returns an error message if no @tool decorator is found, else ``None``.
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                # Handle @tool and @tool(...)
                dec_name = None
                if isinstance(dec, ast.Name):
                    dec_name = dec.id
                elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                    dec_name = dec.func.id
                if dec_name == "tool":
                    return None
    return (
        "No @tool decorator found. Tool modules must contain at least one "
        "function decorated with @tool from pipeline.tools.registry."
    )


def _check_dangerous_imports(code: str) -> str | None:
    """Check for dangerous imports/calls in authored tool code.

    Returns an error message if dangerous patterns are found, else ``None``.
    Blocks: os.system, subprocess.Popen, eval, exec, __import__, compile.
    Allows: subprocess.run (used by check_shapes.py for sandboxed validation).
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        # Check for bare dangerous names: eval(...), exec(...)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _DANGEROUS_PATTERNS:
                return f"Dangerous call found: {node.func.id}() at line {node.lineno}"
            # Check for dotted calls: os.system(...), subprocess.Popen(...)
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    dotted = f"{node.func.value.id}.{node.func.attr}"
                    if dotted in _DANGEROUS_PATTERNS:
                        return (
                            f"Dangerous call found: {dotted}() at line {node.lineno}"
                        )
    return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Write a new tool module into pipeline/tools/. "
        "The file must contain at least one function decorated with @tool. "
        "Validated for syntax, @tool decorator presence, and dangerous patterns. "
        "filename: e.g. 'my_analysis_tool.py'. "
        "code: the full Python source code."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": (
                    "Filename for the tool module (e.g. 'custom_metric.py'). "
                    "Written to pipeline/tools/."
                ),
            },
            "code": {
                "type": "string",
                "description": "Full Python source code for the tool module.",
            },
        },
        "required": ["filename", "code"],
    },
)
def write_tool(filename: str, code: str) -> str:
    """Write a tool module into pipeline/tools/ with multi-layer validation."""
    try:
        filename = _safe_filename(filename)

        # Layer 1: syntax check
        err = _validate_python(code)
        if err:
            return json.dumps({"error": f"Invalid Python: {err}"})

        # Layer 2: @tool decorator check
        err = _check_tool_decorator(code)
        if err:
            return json.dumps({"error": err})

        # Layer 3: dangerous import/call check
        err = _check_dangerous_imports(code)
        if err:
            return json.dumps({"error": err})

        target = _TOOLS_DIR / filename
        with _tool_write_lock:
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
        "Reload a tool module so that newly written @tool functions become "
        "available in the registry. Call this after write_tool. "
        "module_name: the Python module path, e.g. 'pipeline.tools.my_tool'."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "module_name": {
                "type": "string",
                "description": (
                    "Dotted Python module path (e.g. 'pipeline.tools.custom_metric')."
                ),
            },
        },
        "required": ["module_name"],
    },
)
def reload_tools(module_name: str) -> str:
    """Import or reload a tool module to register new @tool functions."""
    try:
        before = set(all_tool_names())

        if module_name in sys.modules:
            mod = importlib.reload(sys.modules[module_name])
        else:
            mod = importlib.import_module(module_name)

        after = set(all_tool_names())
        new_tools = sorted(after - before)

        return json.dumps({
            "status": "reloaded",
            "module": module_name,
            "module_file": getattr(mod, "__file__", None),
            "new_tools": new_tools,
            "total_tools": len(after),
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Validate a registered tool by invoking it in a subprocess with test args. "
        "Checks that the tool runs without crashing and returns valid JSON. "
        "tool_name: name of a registered tool. "
        "test_args: JSON string of keyword arguments to pass."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "Name of the registered tool to validate.",
            },
            "test_args": {
                "type": "string",
                "description": (
                    "JSON string of keyword arguments to pass to the tool, "
                    "e.g. '{\"path\": \"test.txt\"}'."
                ),
            },
        },
        "required": ["tool_name", "test_args"],
    },
)
def validate_tool(tool_name: str, test_args: str) -> str:
    """Validate a registered tool by running it in a subprocess with test args."""
    try:
        td = get_tool(tool_name)
        if td is None:
            return json.dumps({"error": f"Tool not found in registry: {tool_name}"})

        # Parse test args
        try:
            args_dict: dict[str, Any] = json.loads(test_args) if test_args else {}
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Invalid test_args JSON: {exc}"})

        # Build a validation script that imports and calls the tool
        args_repr = json.dumps(args_dict)
        script = textwrap.dedent(f"""\
            import json, sys, traceback
            try:
                from pipeline.tools.registry import get_tool
                # Ensure tool modules are imported
                import pipeline.tools  # noqa: F401
                for attr in dir(pipeline.tools):
                    pass
                td = get_tool({tool_name!r})
                if td is None:
                    print(json.dumps({{"error": "Tool not found after import"}}))
                    sys.exit(0)
                args = json.loads({args_repr!r})
                result = td.func(**args)
                # Check result type
                if not isinstance(result, str):
                    result = str(result)
                # Try to parse as JSON
                try:
                    parsed = json.loads(result)
                    is_json = True
                except (json.JSONDecodeError, TypeError):
                    is_json = False
                print(json.dumps({{
                    "valid": True,
                    "returns_json": is_json,
                    "output_preview": result[:500],
                }}))
            except Exception:
                print(json.dumps({{
                    "valid": False,
                    "error": traceback.format_exc(),
                }}))
        """)

        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(Path.cwd()),
        )

        if proc.returncode != 0:
            return json.dumps({
                "valid": False,
                "error": f"Process exited with code {proc.returncode}",
                "stderr": proc.stderr[:1000],
                "stdout": proc.stdout[:1000],
            })

        return proc.stdout.strip() or json.dumps({"valid": False, "error": "No output"})
    except subprocess.TimeoutExpired:
        return json.dumps({"valid": False, "error": "Validation timed out after 30s"})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Read an existing tool module from pipeline/tools/ to study its pattern. "
        "filename: e.g. 'research_tools.py' or 'market_data.py'."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Filename in pipeline/tools/ (e.g. 'research_tools.py').",
            },
        },
        "required": ["filename"],
    },
)
def read_tool(filename: str) -> str:
    """Read a tool source file for reference."""
    try:
        filename = _safe_filename(filename)
        target = _TOOLS_DIR / filename
        if not target.exists():
            return json.dumps({"error": f"File not found: {target}"})
        content = target.read_text(encoding="utf-8")
        if len(content) > 30_000:
            content = content[:30_000] + f"\n\n... [TRUNCATED — {len(content)} total chars]"
        return content
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description="List all tool module files in pipeline/tools/.",
)
def list_tool_files() -> str:
    """List Python files in pipeline/tools/."""
    try:
        if not _TOOLS_DIR.exists():
            return json.dumps({"error": f"Directory not found: {_TOOLS_DIR}"})
        files = sorted(
            f.name for f in _TOOLS_DIR.iterdir()
            if f.is_file() and f.suffix == ".py" and f.name != "__init__.py"
        )
        return json.dumps({"tools_dir": str(_TOOLS_DIR), "files": files})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Describe a registered tool — returns its name, description, and "
        "parameter schema. tool_name: name of the registered tool."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "Name of the registered tool to describe.",
            },
        },
        "required": ["tool_name"],
    },
)
def describe_tool(tool_name: str) -> str:
    """Return the schema and description for a registered tool."""
    try:
        td = get_tool(tool_name)
        if td is None:
            return json.dumps({"error": f"Tool not found: {tool_name}"})
        return json.dumps({
            "name": td.name,
            "description": td.description,
            "parameters_schema": td.parameters_schema,
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
