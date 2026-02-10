"""
Tool registry — manages tool discovery, schema generation, and dynamic injection.

Each tool is a plain function decorated with ``@tool``.  The decorator records
metadata (name, description, parameter JSON-schema) which the registry uses to
build the OpenAI ``tools`` array and dispatch calls at runtime.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints


@dataclass
class ToolDef:
    """Metadata for a registered tool."""

    name: str
    description: str
    func: Callable[..., Any]
    parameters_schema: dict[str, Any]


# Global registry
_TOOLS: dict[str, ToolDef] = {}


def tool(
    name: str | None = None,
    description: str | None = None,
    parameters_schema: dict[str, Any] | None = None,
):
    """Decorator to register a function as a tool.

    Usage::

        @tool(description="Write content to a file at the given path.")
        def write_file(path: str, content: str) -> str:
            ...
    """

    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip().split("\n")[0]
        schema = parameters_schema or _infer_schema(func)
        _TOOLS[tool_name] = ToolDef(
            name=tool_name,
            description=tool_desc,
            func=func,
            parameters_schema=schema,
        )
        return func

    return decorator


def get_tool(name: str) -> ToolDef | None:
    return _TOOLS.get(name)


def get_tools(*names: str) -> dict[str, Callable]:
    """Return a name→callable dict for the requested tools."""
    result = {}
    for n in names:
        td = _TOOLS.get(n)
        if td:
            result[n] = td.func
    return result


def get_schemas(*names: str) -> list[dict[str, Any]]:
    """Return OpenAI-format tool schemas for the requested tools."""
    schemas = []
    for n in names:
        td = _TOOLS.get(n)
        if td:
            schemas.append({
                "type": "function",
                "function": {
                    "name": td.name,
                    "description": td.description,
                    "parameters": td.parameters_schema,
                },
            })
    return schemas


def all_tool_names() -> list[str]:
    return list(_TOOLS.keys())


def build_toolset(*names: str) -> tuple[dict[str, Callable], list[dict[str, Any]]]:
    """Convenience: return (tools_dict, schemas_list) for the given names."""
    return get_tools(*names), get_schemas(*names)


# ---------------------------------------------------------------------------
# Schema inference from type hints
# ---------------------------------------------------------------------------

_PY_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _infer_schema(func: Callable) -> dict[str, Any]:
    """Best-effort JSON-schema inference from function signature + type hints."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        hint = hints.get(param_name, str)
        json_type = _PY_TO_JSON_TYPE.get(hint, "string")
        properties[param_name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
