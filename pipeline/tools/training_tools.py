"""
Training tools — wrappers for launching, monitoring, and retrieving training
jobs on Basilica compute or local GPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from config import WORKSPACE_DIR
from pipeline.tools.registry import tool


@tool(description="Run a training script locally in a subprocess. Returns stdout/stderr.")
def run_training_local(script_path: str, args: str = "") -> str:
    """Execute a training script in the workspace."""
    full_path = (WORKSPACE_DIR / script_path).resolve()
    if not full_path.exists():
        return json.dumps({"error": f"Script not found: {script_path}"})

    cmd = [sys.executable, str(full_path)]
    if args:
        cmd.extend(args.split())

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(WORKSPACE_DIR),
        )
        output = {
            "returncode": proc.returncode,
            "stdout": proc.stdout[-5000:] if len(proc.stdout) > 5000 else proc.stdout,
            "stderr": proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr,
        }
        return json.dumps(output)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Training timed out after 600s"})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(description="Run a Python snippet in the workspace and return its output. Useful for quick experiments.")
def run_python(code: str) -> str:
    """Execute arbitrary Python code in a subprocess."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(WORKSPACE_DIR),
        )
        output = {
            "returncode": proc.returncode,
            "stdout": proc.stdout[-5000:] if len(proc.stdout) > 5000 else proc.stdout,
            "stderr": proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr,
        }
        return json.dumps(output)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Code execution timed out after 120s"})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(description="Submit a training job to Basilica compute. Returns a job ID for tracking.")
def submit_basilica_job(
    script_path: str,
    gpu_type: str = "A100",
    num_gpus: int = 1,
    docker_image: str = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
) -> str:
    """Submit a training job to Basilica decentralized compute.

    This is a placeholder — actual Basilica integration requires their SDK
    and a funded wallet.
    """
    full_path = (WORKSPACE_DIR / script_path).resolve()
    if not full_path.exists():
        return json.dumps({"error": f"Script not found: {script_path}"})

    # Read the script content to include in the job payload
    script_content = full_path.read_text(encoding="utf-8")

    job_spec = {
        "status": "prepared",
        "note": "Basilica job submission requires BASILICA_API_KEY and a funded wallet.",
        "job_spec": {
            "script": script_path,
            "gpu_type": gpu_type,
            "num_gpus": num_gpus,
            "docker_image": docker_image,
            "script_size_bytes": len(script_content),
        },
    }
    return json.dumps(job_spec, indent=2)
