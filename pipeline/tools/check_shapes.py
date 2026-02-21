"""
Shape-checking tool — dynamically injected into agents that work with
numerical model code (CodeChecker, Debugger).

Runs a candidate model file in a subprocess and verifies that its output
tensors/arrays have the expected shapes for SN50 submission.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

from config import (
    SN50_ASSETS,
    SN50_HORIZON_MINUTES,
    SN50_NUM_PATHS,
    SN50_STEP_MINUTES,
    WORKSPACE_DIR,
)
from pipeline.tools.registry import tool

# Expected number of time steps (including t0)
_EXPECTED_STEPS = (SN50_HORIZON_MINUTES // SN50_STEP_MINUTES) + 1  # 289


def _build_check_script(model_path: str) -> str:
    """Generate a Python script that imports the candidate model and checks shapes."""
    return textwrap.dedent(f"""\
        import json, sys, importlib.util, traceback

        spec = importlib.util.spec_from_file_location("candidate", {model_path!r})
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            print(json.dumps({{"error": traceback.format_exc()}}))
            sys.exit(0)

        # The candidate module must expose a generate_paths(asset, num_paths, num_steps) function
        if not hasattr(mod, "generate_paths"):
            print(json.dumps({{"error": "Module missing 'generate_paths' function"}}))
            sys.exit(0)

        results = {{}}
        assets = {list(SN50_ASSETS.keys())!r}
        num_paths = {SN50_NUM_PATHS}
        num_steps = {_EXPECTED_STEPS}

        for asset in assets:
            try:
                paths = mod.generate_paths(asset, num_paths, num_steps)
                import numpy as np
                arr = np.asarray(paths)
                results[asset] = {{
                    "shape": list(arr.shape),
                    "expected": [num_paths, num_steps],
                    "ok": list(arr.shape) == [num_paths, num_steps],
                    "dtype": str(arr.dtype),
                    "has_nan": bool(np.isnan(arr).any()),
                    "has_inf": bool(np.isinf(arr).any()),
                    "min": float(np.nanmin(arr)),
                    "max": float(np.nanmax(arr)),
                }}
            except Exception:
                results[asset] = {{"error": traceback.format_exc()}}

        print(json.dumps(results, indent=2))
    """)


@tool(
    description=(
        "Run shape validation on a candidate model file. "
        "Checks that generate_paths() returns arrays of shape "
        f"({SN50_NUM_PATHS}, {_EXPECTED_STEPS}) for each SN50 asset. "
        "Also checks for NaN/Inf and reports value ranges."
    ),
)
def check_shapes(model_path: str) -> str:
    """Validate output shapes of a candidate model file."""
    full_path = (WORKSPACE_DIR / model_path).resolve()
    if not full_path.exists():
        return json.dumps({"error": f"Model file not found: {model_path}"})

    script = _build_check_script(str(full_path))
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(WORKSPACE_DIR),
        )
        if proc.returncode != 0:
            return json.dumps({
                "error": (
                    f"Script failed:\nstdout: {proc.stdout}"
                    f"\nstderr: {proc.stderr}"
                ),
            })
        return proc.stdout
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Shape check timed out after 120s"})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
