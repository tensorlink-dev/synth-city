"""
Training tools — local script execution and Basilica GPU cloud rentals.

Local tools (``run_training_local``, ``run_python``) execute scripts in the
workspace.  Basilica tools talk to the secure-cloud GPU marketplace via
``basilica-sdk`` and are budget-capped to cheap offerings only.

Remote training on Basilica
-----------------------------
The recommended flow for GPU-heavy training is:

1. ``rent_cheapest_gpu()`` — provision a GPU pod
2. ``setup_basilica_pod(rental_id)`` — install open-synth-miner + deps
3. ``run_experiment_on_basilica(rental_id, experiment, ...)`` — train on the pod,
   data is downloaded from HuggingFace directly on the pod (not transferred locally)
4. ``stop_gpu_rental(rental_id)`` — release the pod

This avoids running heavy training on the local controller machine.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time

from config import (
    HF_TRAINING_DATA_REPO,
    HF_TOKEN,
    SN50_TO_HF_ASSET,
    TIMEFRAME_CONFIGS,
    WORKSPACE_DIR,
)
from pipeline.tools.registry import tool

# Default SSH key — can be overridden per call
_DEFAULT_SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")

# Timeout for SSH commands during pod setup (seconds)
_SETUP_TIMEOUT = 600
# Timeout for a single training run on the pod (seconds) — 2 h
_TRAIN_TIMEOUT = 7200

# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------


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


@tool(
    description="Run a Python snippet in the workspace and return its output. "
    "Useful for quick experiments.",
)
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


# ---------------------------------------------------------------------------
# Basilica GPU cloud
# ---------------------------------------------------------------------------

def _get_gpu_client():
    """Lazy-init the BasilicaGPUClient (avoids import-time API key check)."""
    from compute.basilica import BasilicaGPUClient
    return BasilicaGPUClient()


@tool(
    description=(
        "List available cheap GPU offerings from Basilica secure cloud. "
        "Shows GPU type, provider, region, hourly rate, spot status, and offering ID. "
        "Only shows GPUs within the configured budget cap."
    ),
)
def list_available_gpus() -> str:
    """List budget-filtered GPU offerings sorted by price (cheapest first)."""
    try:
        client = _get_gpu_client()
        offerings = client.list_cheap_gpus()
        if not offerings:
            return json.dumps({
                "offerings": [],
                "note": "No GPU offerings available within budget. Check BASILICA_MAX_HOURLY_RATE.",
            })
        rows = []
        for o in offerings:
            rows.append({
                "id": o.id,
                "gpu_type": o.gpu_type,
                "gpu_count": o.gpu_count,
                "provider": o.provider,
                "region": o.region,
                "hourly_rate": o.hourly_rate,
                "is_spot": o.is_spot,
                "vcpu": o.vcpu_count,
                "ram_gb": o.system_memory_gb,
                "storage_gb": o.storage_gb,
            })
        return json.dumps({"offerings": rows, "count": len(rows)}, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Register an SSH public key with Basilica for GPU pod access. "
        "Idempotent — if a key is already registered it returns the existing key ID. "
        "Defaults to ~/.ssh/id_ed25519.pub when no path is provided. "
        "Must be called at least once before renting GPU pods."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Label for the key in Basilica (default: synth-city)",
            },
            "public_key_path": {
                "type": "string",
                "description": "Path to SSH public key file (default: ~/.ssh/id_ed25519.pub)",
            },
        },
        "required": [],
    },
)
def register_ssh_key(name: str = "synth-city", public_key_path: str = "") -> str:
    """Register an SSH public key with Basilica (idempotent)."""
    try:
        client = _get_gpu_client()
        key_id = client.ensure_ssh_key(
            name=name,
            public_key_path=public_key_path or None,
        )
        return json.dumps({"status": "ok", "key_id": key_id})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Rent a specific GPU offering by its offering ID. "
        "Returns rental ID, SSH command, IP address, and hourly cost. "
        "Use list_available_gpus first to find offering IDs. "
        "SSH key registration is handled automatically."
    ),
)
def rent_gpu(offering_id: str) -> str:
    """Start a secure-cloud GPU rental for the given offering."""
    try:
        client = _get_gpu_client()
        ssh_key_id = client.ensure_ssh_key()
        resp = client.rent_gpu(offering_id, ssh_public_key_id=ssh_key_id)
        return json.dumps({
            "rental_id": resp.rental_id,
            "status": resp.status,
            "provider": resp.provider,
            "hourly_cost": resp.hourly_cost,
            "ip_address": resp.ip_address,
            "ssh_command": resp.ssh_command,
            "is_spot": resp.is_spot,
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Rent the cheapest available GPU within the budget cap. "
        "Automatically picks the lowest-priced offering. "
        "Returns rental ID, SSH command, IP address, and hourly cost. "
        "SSH key registration is handled automatically."
    ),
)
def rent_cheapest_gpu() -> str:
    """One-click: rent the cheapest GPU currently available."""
    try:
        client = _get_gpu_client()
        ssh_key_id = client.ensure_ssh_key()
        resp = client.rent_cheapest(ssh_public_key_id=ssh_key_id)
        return json.dumps({
            "rental_id": resp.rental_id,
            "status": resp.status,
            "provider": resp.provider,
            "hourly_cost": resp.hourly_cost,
            "ip_address": resp.ip_address,
            "ssh_command": resp.ssh_command,
            "is_spot": resp.is_spot,
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(description="List all active Basilica GPU rentals with status and cost info.")
def list_active_rentals() -> str:
    """Show all current secure-cloud rentals."""
    try:
        client = _get_gpu_client()
        rentals = client.list_active_rentals()
        return json.dumps({"rentals": rentals, "count": len(rentals)}, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Stop a Basilica GPU rental by rental ID. "
        "Returns duration and total cost."
    ),
)
def stop_gpu_rental(rental_id: str) -> str:
    """Stop a running GPU rental."""
    try:
        client = _get_gpu_client()
        result = client.stop_rental(rental_id)
        return json.dumps(result, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(description="Check Basilica account balance.")
def check_gpu_balance() -> str:
    """Return the current account balance."""
    try:
        client = _get_gpu_client()
        balance = client.get_balance()
        return json.dumps(balance, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------------
# Remote training on Basilica pods
# ---------------------------------------------------------------------------

def _ssh_run(
    host: str,
    port: int,
    user: str,
    command: str,
    key_path: str = _DEFAULT_SSH_KEY,
    timeout: int = _SETUP_TIMEOUT,
    stdin_data: str | None = None,
) -> tuple[int, str, str]:
    """Run *command* on a remote host via SSH.

    Returns ``(returncode, stdout, stderr)``.
    """
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=30",
        "-p", str(port),
    ]
    if key_path and os.path.exists(key_path):
        ssh_cmd += ["-i", key_path]
    ssh_cmd += [f"{user}@{host}", command]

    proc = subprocess.run(
        ssh_cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        input=stdin_data,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _parse_ssh_command(ssh_command: str) -> dict:
    """Parse a Basilica SSH command string into ``{host, port, user}``.

    Basilica returns SSH access as a plain string, e.g.::

        "ssh ubuntu@192.0.2.1 -p 22022"
        "ssh -p 22022 ubuntu@192.0.2.1"

    Returns a dict with keys ``host`` (str), ``port`` (int), ``user`` (str).
    """
    user_host = re.search(r"([A-Za-z0-9_]+)@([\w.\-]+)", ssh_command)
    if not user_host:
        raise RuntimeError(
            f"Cannot parse user@host from Basilica SSH command: {ssh_command!r}"
        )
    user, host = user_host.groups()
    port_m = re.search(r"-p\s+(\d+)", ssh_command)
    port = int(port_m.group(1)) if port_m else 22
    return {"host": host, "port": port, "user": user}


def _get_ssh_creds(rental_id: str) -> dict:
    """Return ``{host, port, user}`` for *rental_id* or raise RuntimeError.

    Calls ``get_rental_status()`` (which internally uses
    ``list_secure_cloud_rentals``) and parses the ``ssh_command`` string
    that Basilica returns.
    """
    client = _get_gpu_client()
    status = client.get_rental_status(rental_id)
    ssh_cmd = status.get("ssh_command")
    if not ssh_cmd:
        raise RuntimeError(
            f"No SSH command for rental {rental_id!r} "
            f"(status={status.get('status')!r}). "
            "The pod may still be provisioning — wait a moment and try again."
        )
    return _parse_ssh_command(ssh_cmd)


@tool(
    description=(
        "Wait for a Basilica rental to become SSH-accessible, then install "
        "open-synth-miner and all training dependencies on the pod. "
        "Must be called before run_experiment_on_basilica. "
        "rental_id: the rental returned by rent_gpu / rent_cheapest_gpu. "
        "ssh_key_path: path to your SSH private key (default: ~/.ssh/id_ed25519). "
        "osm_install: pip install spec for open-synth-miner "
        "(default: 'open-synth-miner'; use 'git+https://github.com/tensorlink-dev/open-synth-miner' "
        "to install directly from GitHub)."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "rental_id": {"type": "string", "description": "Basilica rental ID"},
            "ssh_key_path": {
                "type": "string",
                "description": "Path to SSH private key (default: ~/.ssh/id_ed25519)",
            },
            "osm_install": {
                "type": "string",
                "description": "pip install spec for open-synth-miner",
            },
        },
        "required": ["rental_id"],
    },
)
def setup_basilica_pod(
    rental_id: str,
    ssh_key_path: str = "",
    osm_install: str = "open-synth-miner",
) -> str:
    """Install open-synth-miner + deps on a Basilica pod and configure HF token."""
    key = ssh_key_path or _DEFAULT_SSH_KEY
    try:
        # Poll for SSH readiness (up to 5 minutes)
        creds = None
        deadline = time.time() + 300
        last_err = ""
        while time.time() < deadline:
            try:
                creds = _get_ssh_creds(rental_id)
                rc, _, err = _ssh_run(
                    creds["host"], creds["port"], creds["user"],
                    "echo ready", key_path=key, timeout=15,
                )
                if rc == 0:
                    break
                last_err = err
            except Exception as e:
                last_err = str(e)
            time.sleep(10)
        else:
            return json.dumps({
                "error": f"Pod did not become SSH-accessible within 5 minutes. Last error: {last_err}"
            })

        host, port, user = creds["host"], creds["port"], creds["user"]

        # Build setup commands
        hf_token_export = f'export HF_TOKEN="{HF_TOKEN}"' if HF_TOKEN else ""
        setup_script = (
            f"set -e\n"
            f"pip install --quiet --upgrade pip\n"
            f"pip install --quiet {shlex.quote(osm_install)}\n"
            f"pip install --quiet huggingface_hub pyarrow pandas numpy torch\n"
        )
        if hf_token_export:
            setup_script += (
                f"{hf_token_export}\n"
                f'echo "export HF_TOKEN={HF_TOKEN}" >> ~/.bashrc\n'
            )

        rc, stdout, stderr = _ssh_run(
            host, port, user, f"bash -s", key_path=key,
            timeout=_SETUP_TIMEOUT, stdin_data=setup_script,
        )
        if rc != 0:
            return json.dumps({
                "error": f"Pod setup failed (exit {rc})",
                "stderr": stderr[-3000:],
                "stdout": stdout[-1000:],
            })

        return json.dumps({
            "status": "ready",
            "rental_id": rental_id,
            "host": host,
            "user": user,
            "hf_token_configured": bool(HF_TOKEN),
            "osm_installed": osm_install,
            "stdout": stdout[-1000:],
        }, indent=2)

    except subprocess.TimeoutExpired:
        return json.dumps({"error": "SSH timed out during pod setup."})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Run an experiment on a Basilica GPU pod via SSH. "
        "The pod downloads training data directly from HuggingFace — "
        "no local data transfer needed. "
        "Returns CRPS metrics identical to run_experiment. "
        "rental_id: Basilica rental (must have been set up with setup_basilica_pod). "
        "experiment: experiment config JSON from create_experiment. "
        "epochs: training epochs. "
        "timeframe: '5m' or '1m' — selects HF dataset and prediction horizon. "
        "ssh_key_path: path to SSH private key (default: ~/.ssh/id_ed25519)."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "rental_id": {"type": "string", "description": "Basilica rental ID"},
            "experiment": {"type": "string", "description": "Experiment config JSON"},
            "epochs": {"type": "integer", "description": "Training epochs (default: 1)"},
            "timeframe": {
                "type": "string",
                "enum": ["5m", "1m"],
                "description": "Data timeframe: '5m' (288-step) or '1m' (60-step)",
            },
            "ssh_key_path": {
                "type": "string",
                "description": "Path to SSH private key (default: ~/.ssh/id_ed25519)",
            },
        },
        "required": ["rental_id", "experiment"],
    },
)
def run_experiment_on_basilica(
    rental_id: str,
    experiment: str,
    epochs: int = 1,
    timeframe: str = "5m",
    ssh_key_path: str = "",
) -> str:
    """Run a training experiment on a Basilica GPU pod and return metrics."""
    key = ssh_key_path or _DEFAULT_SSH_KEY
    try:
        creds = _get_ssh_creds(rental_id)
        host, port, user = creds["host"], creds["port"], creds["user"]

        exp_dict = json.loads(experiment) if isinstance(experiment, str) else experiment
        # Strip timeframe tag — the pod will use the explicit timeframe arg
        exp_dict.pop("timeframe", None)

        tf_cfg = TIMEFRAME_CONFIGS.get(timeframe, TIMEFRAME_CONFIGS["5m"])
        asset_files = {
            hf_name: f"data/{hf_name}/{tf_cfg['file_suffix']}"
            for hf_name in SN50_TO_HF_ASSET.values()
        }

        # Self-contained Python script that runs entirely on the pod.
        # Only imports open-synth-miner (must be pre-installed via setup_basilica_pod).
        train_script = f"""\
import json, os, sys

os.environ.setdefault("HF_TOKEN", {HF_TOKEN!r})

experiment = {json.dumps(exp_dict)}
epochs     = {epochs}
hf_repo    = {HF_TRAINING_DATA_REPO!r}
asset_files = {json.dumps(asset_files)}
input_len  = {int(tf_cfg["input_len"])}
pred_len   = {int(tf_cfg["pred_len"])}

try:
    import importlib as _il
    _session_cls = None
    for _mod_path in ("src.research.agent_api", "research.agent_api"):
        try:
            _mod = _il.import_module(_mod_path)
            _session_cls = getattr(_mod, "ResearchSession", None)
            if _session_cls is not None:
                break
        except ImportError:
            pass
    if _session_cls is None:
        raise ImportError(
            "Cannot import ResearchSession. "
            "Install open-synth-miner: pip install open-synth-miner"
        )
    ResearchSession = _session_cls
    session = ResearchSession()

    try:
        from src.data.market_data_loader import (
            HFOHLCVSource, MarketDataLoader, ZScoreEngineer,
        )
        source = HFOHLCVSource(
            repo_id=hf_repo,
            asset_files=asset_files,
            repo_type="dataset",
        )
        loader = MarketDataLoader(
            data_source=source,
            engineer=ZScoreEngineer(),
            assets=list(asset_files.keys()),
            input_len=input_len,
            pred_len=pred_len,
            batch_size=64,
            feature_dim=4,
            gap_handling="ffill",
            stride=12,
        )
        result = session.run(experiment, epochs=epochs, data_loader=loader)
    except TypeError:
        # ResearchSession.run() may not accept data_loader yet
        result = session.run(experiment, epochs=epochs)

    print(json.dumps(result, default=str))

except Exception as exc:
    import traceback
    print(json.dumps({{
        "status": "error",
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }}, default=str))
    sys.exit(1)
"""

        rc, stdout, stderr = _ssh_run(
            host, port, user,
            f"python3 -c {shlex.quote(train_script)}",
            key_path=key,
            timeout=_TRAIN_TIMEOUT,
        )

        if rc != 0:
            return json.dumps({
                "status": "error",
                "error": f"Remote training script exited with code {rc}",
                "stderr": stderr[-3000:],
                "stdout": stdout[-1000:],
            })

        # Parse the JSON printed by the script
        output = stdout.strip()
        try:
            return output  # Already valid JSON from the remote script
        except Exception:
            return json.dumps({"status": "error", "raw_output": output[:3000]})

    except subprocess.TimeoutExpired:
        return json.dumps({
            "status": "error",
            "error": f"Remote training timed out after {_TRAIN_TIMEOUT}s.",
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
