"""
Component registration tools — write new blocks/heads into open-synth-miner
and trigger re-discovery so they appear in the registry immediately.

These tools give the ComponentAuthor agent the ability to create custom
PyTorch components that plug into the open-synth-miner framework.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

from pipeline.tools.registry import tool

logger = logging.getLogger(__name__)

# open-synth-miner component directories (relative to repo root)
_COMPONENTS_DIR = Path("src/models/components")
_CONFIGS_DIR = Path("configs/model")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@tool(
    description=(
        "Write a new PyTorch block or head into open-synth-miner's component directory. "
        "The file is placed in src/models/components/ and auto-discovered by the registry. "
        "filename: e.g. 'my_block.py'. "
        "code: the full Python source code for the component."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Filename (e.g. 'wavelet_block.py'). Written to src/models/components/.",
            },
            "code": {
                "type": "string",
                "description": "Full Python source code for the component.",
            },
        },
        "required": ["filename", "code"],
    },
)
def write_component(filename: str, code: str) -> str:
    """Write a component file into the open-synth-miner components directory."""
    try:
        if not filename.endswith(".py"):
            filename += ".py"

        target = _COMPONENTS_DIR / filename
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
        "Read an existing component file from open-synth-miner to study its structure. "
        "path: relative to repo root, e.g. 'src/models/components/transformer.py'."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path relative to repo root.",
            },
        },
        "required": ["path"],
    },
)
def read_component(path: str) -> str:
    """Read a component source file for reference."""
    try:
        target = Path(path)
        if not target.exists():
            return json.dumps({"error": f"File not found: {path}"})
        content = target.read_text(encoding="utf-8")
        if len(content) > 30_000:
            content = content[:30_000] + f"\n\n... [TRUNCATED — {len(content)} total chars]"
        return content
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "List all files in open-synth-miner's component directory to see what's registered."
    ),
)
def list_component_files() -> str:
    """List files in src/models/components/."""
    try:
        if not _COMPONENTS_DIR.exists():
            return json.dumps({"error": f"Directory not found: {_COMPONENTS_DIR}"})
        files = sorted(f.name for f in _COMPONENTS_DIR.iterdir() if f.is_file() and f.suffix == ".py")
        return json.dumps({"components_dir": str(_COMPONENTS_DIR), "files": files})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Trigger re-discovery of components so newly written blocks/heads appear in the registry. "
        "Call this after write_component to make the new component available to list_blocks/list_heads."
    ),
)
def reload_registry() -> str:
    """Re-discover components from disk and refresh the registry."""
    try:
        from src.models.registry import discover_components, registry

        discover_components(str(_COMPONENTS_DIR))
        block_count = len(registry.list_blocks()) if hasattr(registry, "list_blocks") else "unknown"
        head_count = len(registry.list_heads()) if hasattr(registry, "list_heads") else "unknown"

        return json.dumps({
            "status": "reloaded",
            "blocks": block_count,
            "heads": head_count,
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


@tool(
    description=(
        "Write a YAML model config / hybrid recipe to configs/model/. "
        "filename: e.g. 'my_hybrid.yaml'. "
        "content: the YAML config content."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "YAML filename (e.g. 'wavelet_hybrid.yaml').",
            },
            "content": {
                "type": "string",
                "description": "Full YAML config content.",
            },
        },
        "required": ["filename", "content"],
    },
)
def write_config(filename: str, content: str) -> str:
    """Write a YAML model config into configs/model/."""
    try:
        if not filename.endswith(".yaml") and not filename.endswith(".yml"):
            filename += ".yaml"

        target = _CONFIGS_DIR / filename
        _ensure_dir(target.parent)
        target.write_text(content, encoding="utf-8")

        return json.dumps({
            "status": "written",
            "path": str(target),
            "bytes": len(content),
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
