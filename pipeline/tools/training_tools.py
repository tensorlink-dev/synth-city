"""
Training tools — local script execution and Basilica GPU cloud deployments.

Local tools (``run_training_local``, ``run_python``) execute scripts in the
workspace.  Basilica tools talk to the GPU marketplace via ``basilica-sdk``.

Remote training on Basilica (Docker image deployments)
--------------------------------------------------------
The recommended flow for GPU-heavy training is:

1. ``create_training_deployment()`` — spin up a Docker-image-based GPU pod
2. ``run_experiment_on_deployment(url, experiment)`` — train via HTTP
3. ``delete_training_deployment(name)`` — free GPU resources

The deployment uses a pre-built Docker image with open-synth-miner already
installed.  No SSH or pip install needed.  Training data is downloaded from
HuggingFace directly on the pod.

Legacy SSH rental tools (``rent_cheapest_gpu``, ``setup_basilica_pod``,
``run_experiment_on_basilica``) are still registered for direct/CLI use but
are not exposed to pipeline agents.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from typing import Any

from config import (
    BASILICA_DEPLOY_IMAGE,
    HF_TOKEN,
    HF_TRAINING_DATA_REPO,
    SN50_TO_HF_ASSET,
    TIMEFRAME_CONFIGS,
    WORKSPACE_DIR,
)
from pipeline.monitor import get_monitor
from pipeline.tools.registry import tool

logger = logging.getLogger(__name__)
_mon = get_monitor()

# Default SSH key — can be overridden per call
_DEFAULT_SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")

# Timeout for SSH commands during pod setup (seconds)
_SETUP_TIMEOUT = 600
# Timeout for a single training run on the pod (seconds) — 2 h
_TRAIN_TIMEOUT = 7200

# Regex for benign nvidia-smi warnings that do not indicate GPU problems.
# infoROM corruption is a metadata-only issue and does not affect compute.
_BENIGN_GPU_WARNING_RE = re.compile(
    r"^WARNING:\s*infoROM is corrupted at gpu \S+\s*$", re.MULTILINE,
)


def _fix_pip_install_spec(spec: str) -> str:
    """Fix a common LLM mistake: ``git+URL[extras]`` is invalid pip syntax.

    pip reads ``[extras]`` as part of the URL, so git tries to clone a
    repository that doesn't exist.  The correct PEP 508 form is::

        package[extras] @ git+URL

    This helper detects the broken pattern and rewrites it.
    """
    m = re.match(r"^(git\+.+?)(\[[^\]]+\])$", spec.strip())
    if not m:
        return spec
    url, extras = m.group(1), m.group(2)
    # Derive the package name from the last path component of the URL,
    # stripping .git suffix and any @ref / #fragment.
    # e.g. git+https://github.com/org/open-synth-miner.git@v1 → open-synth-miner
    #      git+ssh://git@github.com/org/repo.git             → repo
    last_segment = url.rstrip("/").rsplit("/", 1)[-1]
    pkg_name = re.sub(r"(?:\.git)?(?:[@#].*)?$", "", last_segment)
    if not pkg_name:
        return spec
    fixed = f"{pkg_name}{extras} @ {url}"
    logger.info("Rewrote malformed pip spec %r → %r", spec, fixed)
    return fixed


def _clean_gpu_info(raw: str) -> str:
    """Strip benign nvidia-smi warnings that mislead the agent into thinking
    the GPU is broken. The infoROM corruption warning, for example, only means
    the small EEPROM storing card metadata has a bad checksum — compute is
    unaffected."""
    cleaned = _BENIGN_GPU_WARNING_RE.sub("", raw)
    # Collapse runs of blank lines left behind by the removal.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

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
            "stdout": proc.stdout[-20000:] if len(proc.stdout) > 20000 else proc.stdout,
            "stderr": proc.stderr[-15000:] if len(proc.stderr) > 15000 else proc.stderr,
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
            "stdout": proc.stdout[-20000:] if len(proc.stdout) > 20000 else proc.stdout,
            "stderr": proc.stderr[-15000:] if len(proc.stderr) > 15000 else proc.stderr,
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
                "available": getattr(o, "availability", None),
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
        "Verifies the registered key matches the local key — if they differ, "
        "the stale remote key is replaced automatically. "
        "Generates a new ed25519 keypair if no local key exists. "
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
    """Register an SSH public key with Basilica (idempotent, verifies key match)."""
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
        "SSH key registration is handled automatically. "
        "NOTE: Some offerings (especially Verda) may fail with 'Operating "
        "system is not valid'. If you hit this, use rent_cheapest_gpu instead — "
        "it automatically skips incompatible offerings."
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
        error_msg = str(exc)
        result: dict[str, Any] = {"error": f"{type(exc).__name__}: {exc}"}
        if "Operating system is not valid" in error_msg:
            logger.warning(
                "Offering %s rejected: OS not valid for instance type. "
                "Use rent_cheapest_gpu() to auto-skip incompatible offerings.",
                offering_id,
            )
            result["hint"] = (
                "This offering is incompatible (OS not valid for this instance type). "
                "Use rent_cheapest_gpu instead — it automatically tries multiple "
                "offerings and skips ones that fail."
            )
        return json.dumps(result)


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


@tool(
    description=(
        "Diagnostic tool: check GPU availability on Basilica. "
        "Shows ALL GPUs the platform has (unfiltered) alongside the subset "
        "that pass your budget/type filters. Use this to diagnose whether "
        "deployment failures are caused by GPU availability vs image issues. "
        "If total_all > 0 but filtered_count == 0, your filters are too strict."
    ),
)
def check_gpu_availability() -> str:
    """Compare all available GPUs against the current budget/type filters."""
    try:
        client = _get_gpu_client()
        # Fetch *all* offerings (including unavailable) for the full picture
        all_gpus = client.list_all_gpus(available_only=False)
        filtered_gpus = client.list_cheap_gpus()

        # Summarise all GPUs by type, tracking availability
        type_summary: dict[str, dict[str, Any]] = {}
        for o in all_gpus:
            gpu_type = o.gpu_type or "unknown"
            entry = type_summary.setdefault(
                gpu_type, {"rates": [], "available": 0, "unavailable": 0}
            )
            entry["rates"].append(float(o.hourly_rate))
            if getattr(o, "availability", True):
                entry["available"] += 1
            else:
                entry["unavailable"] += 1

        all_summary = []
        for gpu_type, info in sorted(type_summary.items()):
            rates = info["rates"]
            all_summary.append({
                "gpu_type": gpu_type,
                "total": len(rates),
                "available": info["available"],
                "unavailable": info["unavailable"],
                "min_rate": round(min(rates), 4),
                "max_rate": round(max(rates), 4),
            })

        filtered_rows = []
        for o in filtered_gpus:
            filtered_rows.append({
                "gpu_type": o.gpu_type,
                "hourly_rate": o.hourly_rate,
                "provider": o.provider,
                "region": o.region,
                "is_spot": o.is_spot,
            })

        from config import BASILICA_ALLOWED_GPU_TYPES, BASILICA_MAX_HOURLY_RATE

        total_available = sum(e["available"] for e in type_summary.values())
        total_unavailable = sum(e["unavailable"] for e in type_summary.values())

        result: dict[str, Any] = {
            "filters": {
                "max_hourly_rate": BASILICA_MAX_HOURLY_RATE,
                "allowed_gpu_types": BASILICA_ALLOWED_GPU_TYPES,
            },
            "total_all": len(all_gpus),
            "total_available": total_available,
            "total_unavailable": total_unavailable,
            "all_by_type": all_summary,
            "filtered_count": len(filtered_gpus),
            "filtered": filtered_rows[:20],  # cap output size
        }
        if total_available > 0 and len(filtered_gpus) == 0:
            result["diagnosis"] = (
                f"Platform has {total_available} available GPU(s) but NONE pass "
                "your filters. Loosen BASILICA_MAX_HOURLY_RATE or "
                "BASILICA_ALLOWED_GPU_TYPES."
            )
        elif total_available == 0 and len(all_gpus) > 0:
            result["diagnosis"] = (
                f"Platform lists {len(all_gpus)} GPU(s) but ALL are marked "
                "unavailable. This is a Basilica-side stock issue."
            )
        elif len(all_gpus) == 0:
            result["diagnosis"] = (
                "Platform reports ZERO GPU offerings. "
                "This is a Basilica-side availability issue."
            )
        else:
            result["diagnosis"] = (
                f"{len(filtered_gpus)} GPU(s) available within budget. "
                "Filters look fine."
            )
        return json.dumps(result, indent=2)
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
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=30",
        "-p", str(port),
    ]
    if key_path and os.path.exists(key_path):
        ssh_cmd += ["-i", key_path]
    elif key_path:
        logger.warning("SSH private key not found at %s — connection will likely fail", key_path)
    else:
        logger.warning("No SSH private key path specified — connection will likely fail")
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


_DEFAULT_OSM_INSTALL = (
    "open-synth-miner @ "
    "git+https://github.com/tensorlink-dev/open-synth-miner.git"
)


@tool(
    description=(
        "Wait for a Basilica rental to become SSH-accessible, then install "
        "open-synth-miner and all training dependencies on the pod. "
        "Must be called before run_experiment_on_basilica. "
        "rental_id: the rental returned by rent_gpu / rent_cheapest_gpu. "
        "ssh_key_path: path to your SSH private key "
        "(default: ~/.ssh/id_ed25519). "
        "osm_install: pip install spec for open-synth-miner "
        "(default: install from GitHub)."
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
                "description": (
                    "pip install spec for open-synth-miner. "
                    "Default installs from GitHub. Override "
                    "only if you need a specific branch/fork."
                ),
            },
        },
        "required": ["rental_id"],
    },
)
def setup_basilica_pod(
    rental_id: str,
    ssh_key_path: str = "",
    osm_install: str = "",
) -> str:
    """Install open-synth-miner + deps on a Basilica pod and configure HF token."""
    osm_install = _fix_pip_install_spec(osm_install or _DEFAULT_OSM_INSTALL)
    key = ssh_key_path or _DEFAULT_SSH_KEY
    try:
        # Ensure the Basilica-registered key matches our local key *before*
        # we start polling the pod.  This is the most common cause of
        # "Permission denied (publickey)" failures — a stale key was
        # registered from a previous run or a different machine.
        try:
            client = _get_gpu_client()
            client.ensure_ssh_key(
                public_key_path=key + ".pub" if not key.endswith(".pub") else key,
            )
        except Exception as e:
            logger.warning("Could not verify SSH key registration: %s", e)

        # Poll for SSH readiness (up to 5 minutes)
        creds = None
        deadline = time.time() + 300
        ssh_errors: list[str] = []
        while time.time() < deadline:
            try:
                creds = _get_ssh_creds(rental_id)
                rc, _, err = _ssh_run(
                    creds["host"], creds["port"], creds["user"],
                    "echo ready", key_path=key, timeout=15,
                )
                if rc == 0:
                    break
                ssh_errors.append(err.strip())
            except Exception as e:
                ssh_errors.append(str(e))
            time.sleep(10)
        else:
            # Deduplicate consecutive identical errors for readability
            deduped: list[str] = []
            for e in ssh_errors:
                if not deduped or e != deduped[-1]:
                    deduped.append(e)

            # Add diagnostic info about key mismatch
            key_info = "unknown"
            try:
                key_info = f"local_key_exists={os.path.exists(key)}"
                pub_path = key + ".pub" if not key.endswith(".pub") else key
                key_info += f", local_pub_exists={os.path.exists(pub_path)}"
            except Exception:
                pass

            return json.dumps({
                "error": "Pod did not become SSH-accessible within 5 minutes.",
                "rental_id": rental_id,
                "ssh_attempts": len(ssh_errors),
                "ssh_errors": deduped[-10:],  # last 10 unique errors
                "key_path": key,
                "key_diagnostics": key_info,
                "hint": (
                    "If you see 'Permission denied (publickey)', the registered "
                    "Basilica SSH key may not match your local key. Try calling "
                    "register_ssh_key() to force re-registration."
                ),
            })

        host, port, user = creds["host"], creds["port"], creds["user"]

        # Build setup commands.
        # * ``python3 -m pip`` — ensures packages install for the same
        #   interpreter that ``run_experiment_on_basilica`` invokes.
        # * ``--break-system-packages`` — required on Debian 12+ / Ubuntu
        #   23.04+ which enforce PEP 668 (externally-managed-environment).
        # * Final ``python3 -c …`` — smoke-tests the import so we fail
        #   fast instead of reporting "ready" with a broken environment.
        _pip = "python3 -m pip install --break-system-packages"
        hf_token_export = f'export HF_TOKEN="{HF_TOKEN}"' if HF_TOKEN else ""
        setup_script = (
            f"set -e\n"
            f"{_pip} --quiet --upgrade pip\n"
            f"{_pip} --quiet {shlex.quote(osm_install)}\n"
            f"{_pip} --quiet huggingface_hub pyarrow pandas numpy torch\n"
            f"python3 << 'PYEOF'\n"
            f"tried = ['osa.research.agent_api', 'src.research.agent_api', 'research.agent_api']\n"
            f"for m in tried:\n"
            f"    try:\n"
            f"        __import__(m)\n"
            f"        break\n"
            f"    except ImportError:\n"
            f"        pass\n"
            f"else:\n"
            f"    raise ImportError(\n"
            f"        'ResearchSession not importable after install; tried ' "
            f"+ str(tried))\n"
            f"PYEOF\n"
        )
        if hf_token_export:
            setup_script += (
                f"{hf_token_export}\n"
                f'echo "export HF_TOKEN={HF_TOKEN}" >> ~/.bashrc\n'
            )

        rc, stdout, stderr = _ssh_run(
            host, port, user, "bash -s", key_path=key,
            timeout=_SETUP_TIMEOUT, stdin_data=setup_script,
        )
        if rc != 0:
            logger.error(
                "Pod setup failed for rental %s (exit %d). stderr:\n%s",
                rental_id, rc, stderr[-5000:],
            )
            return json.dumps({
                "error": f"Pod setup failed (exit {rc})",
                "rental_id": rental_id,
                "stderr": stderr[-15000:],
                "stdout": stdout[-10000:],
            })

        result: dict[str, object] = {
            "status": "ready",
            "rental_id": rental_id,
            "host": host,
            "user": user,
            "hf_token_configured": bool(HF_TOKEN),
            "osm_installed": osm_install,
            "stdout": stdout[-5000:],
        }
        if stderr.strip():
            result["stderr"] = stderr[-5000:]
        return json.dumps(result, indent=2)

    except subprocess.TimeoutExpired:
        return json.dumps({
            "error": "SSH timed out during pod setup.",
            "rental_id": rental_id,
        })
    except Exception as exc:
        return json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "rental_id": rental_id,
        })


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
    for _mod_path in ("osa.research.agent_api", "src.research.agent_api", "research.agent_api"):
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
        from osa.data.market_data_loader import (
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
            # Capture GPU diagnostics to help debug OOM / hardware issues
            gpu_info = ""
            try:
                _, gpu_out, _ = _ssh_run(
                    host, port, user, "nvidia-smi", key_path=key, timeout=15,
                )
                gpu_info = _clean_gpu_info(gpu_out)
            except Exception:
                gpu_info = "(nvidia-smi unavailable)"

            logger.error(
                "Remote training failed for rental %s (exit %d). stderr:\n%s",
                rental_id, rc, stderr[-5000:],
            )
            return json.dumps({
                "status": "error",
                "rental_id": rental_id,
                "error": f"Remote training script exited with code {rc}",
                "stderr": stderr[-15000:],
                "stdout": stdout[-10000:],
                "gpu_info": gpu_info[:3000],
            })

        # The script prints a single JSON line to stdout on success.
        # Extract the last non-empty line (ignore pip/import warnings).
        output = stdout.strip()
        last_line = ""
        for line in reversed(output.splitlines()):
            stripped = line.strip()
            if stripped and stripped[0] == "{":
                last_line = stripped
                break
        if last_line:
            try:
                json.loads(last_line)  # validate it's proper JSON
                return last_line
            except json.JSONDecodeError:
                pass
        # Couldn't parse JSON — return raw output for debugging
        return json.dumps({
            "status": "error",
            "rental_id": rental_id,
            "error": "Could not parse JSON from training script output",
            "raw_output": output[-15000:],
        })

    except subprocess.TimeoutExpired:
        # Try to capture partial output and GPU state even after timeout
        gpu_info = ""
        try:
            creds2 = _get_ssh_creds(rental_id)
            _, gpu_out, _ = _ssh_run(
                creds2["host"], creds2["port"], creds2["user"],
                "nvidia-smi", key_path=key, timeout=15,
            )
            gpu_info = _clean_gpu_info(gpu_out)
        except Exception:
            pass
        return json.dumps({
            "status": "error",
            "rental_id": rental_id,
            "error": f"Remote training timed out after {_TRAIN_TIMEOUT}s.",
            "gpu_info": gpu_info[:3000] if gpu_info else "(unavailable)",
        })
    except Exception as exc:
        return json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "rental_id": rental_id,
        })


# ---------------------------------------------------------------------------
# Basilica deployment-based training (Docker image approach)
# ---------------------------------------------------------------------------

# HTTP timeout for training requests to a deployment (seconds)
_DEPLOY_TRAIN_TIMEOUT = 7200

# Retry parameters for transient HTTP errors (5xx) from deployments
_DEPLOY_MAX_RETRIES = 4
_DEPLOY_INITIAL_BACKOFF = 30  # seconds
_DEPLOY_BACKOFF_FACTOR = 2    # exponential multiplier

# Timeout for deployment health/readiness probes (seconds)
_DEPLOY_HEALTH_TIMEOUT = 300
_DEPLOY_HEALTH_INTERVAL = 15
# Abort early if the server returns this many consecutive HTTP 5xx responses
# (meaning the server process is running but broken — waiting longer won't help)
_DEPLOY_HEALTH_MAX_SERVER_ERRORS = 5
# Abort early if DNS/connection failures persist this many consecutive probes
# (meaning the hostname never resolved — the deployment may be deleted or invalid)
_DEPLOY_HEALTH_MAX_DNS_FAILURES = 8
# Abort early if the reverse proxy keeps returning 500s for this many probes.
# Reverse-proxy 500s mean the pod is reachable but the training server inside
# never started (e.g. crash on import, OOM, missing dependency).  Waiting
# longer won't help — the container needs to be deleted and recreated.
_DEPLOY_HEALTH_MAX_PROXY_ERRORS = 12
# How often (in probes) to check the pod phase via the Basilica API
_DEPLOY_HEALTH_PHASE_CHECK_INTERVAL = 4

_DEPLOYMENT_NAME_PREFIX = "synth-city-trainer"

# ---------------------------------------------------------------------------
# Cross-deployment failure tracking
# ---------------------------------------------------------------------------
# Tracks consecutive deployment health-check failures across create/delete
# cycles so the agent gets a clear signal to stop recreating deployments
# when the Docker image itself is broken.
_MAX_CONSECUTIVE_DEPLOY_FAILURES = 3
_deployment_failure_count = 0


def _probe_deployment_health(url: str, share_token: str = "") -> tuple[bool, str]:
    """Send a lightweight GET probe to a deployment's ``/health`` endpoint.

    Returns ``(healthy, detail)`` where *healthy* is True if the training
    server responded with HTTP 200.  The ``/health`` endpoint in
    ``training_server.py`` validates that ``ResearchSession`` is importable,
    so a 200 means the server can actually run experiments (not just that the
    HTTP process is alive).

    A 503 from ``/health`` means the server is up but ResearchSession is
    broken (e.g. open-synth-miner failed to install) — this is treated as
    **unhealthy** because training requests would also fail.
    """
    import urllib.error
    import urllib.request

    probe_url = url.rstrip("/") + "/health"
    if share_token:
        probe_url += f"?token={share_token}"
    try:
        req = urllib.request.Request(probe_url, method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return True, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode()[:500]
        except Exception:
            pass
        if exc.code == 503:
            # Server is up but ResearchSession is broken — not healthy.
            # Tag as [server] so the caller can distinguish from proxy errors.
            return False, f"HTTP 503: server up but unhealthy [server] ({body})"
        if exc.code == 500:
            # Distinguish reverse-proxy 500 from training server 500.
            # Our global exception handler returns JSON with an "error" key;
            # a bare proxy 500 is typically HTML or has an empty body
            # (ingress returns 500 when no backend pod is ready yet).
            if not body.strip() or "<html" in body.lower():
                return False, f"HTTP 500: Internal Server Error [reverse-proxy] ({body})"
            return False, f"HTTP 500: Internal Server Error [server] ({body})"
        if exc.code in (502, 504):
            # 502 Bad Gateway / 504 Gateway Timeout — reverse proxy can't
            # reach the backend container (still starting or crashed).
            return False, f"HTTP {exc.code}: {exc.reason} [reverse-proxy] ({body})"
        # Other 4xx — server is responsive, /health just isn't mapped (older image?)
        if 400 <= exc.code < 500:
            return True, f"HTTP {exc.code} (server responsive)"
        return False, f"HTTP {exc.code}: {exc.reason} ({body})"
    except urllib.error.URLError as exc:
        return False, f"Connection failed: {exc.reason}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


@tool(
    description=(
        "Wait for a Basilica deployment to become healthy and ready to accept "
        "training requests. Probes the deployment URL until the HTTP server "
        "responds. Call this AFTER get_training_deployment shows phase='Running' "
        "and BEFORE sending run_experiment_on_deployment requests. "
        "Returns when the deployment is ready, or an error after timeout."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "deployment_url": {
                "type": "string",
                "description": "Deployment URL (from create_training_deployment)",
            },
            "share_token": {
                "type": "string",
                "description": "Share token for private deployments",
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait (default: 300)",
            },
        },
        "required": ["deployment_url"],
    },
)
def wait_for_deployment_ready(
    deployment_url: str,
    share_token: str = "",
    timeout: int = _DEPLOY_HEALTH_TIMEOUT,
) -> str:
    """Probe a deployment URL until the HTTP server is responsive.

    Includes five early-abort mechanisms so the agent doesn't waste the full
    timeout on obviously-broken deployments:

    1. **Consecutive server errors** — if the health endpoint returns HTTP 5xx
       ``_DEPLOY_HEALTH_MAX_SERVER_ERRORS`` times in a row, the server process
       is up but the health check is crashing.  Waiting longer won't fix it.
    2. **Consecutive DNS failures** — if the hostname never resolves after
       ``_DEPLOY_HEALTH_MAX_DNS_FAILURES`` probes, the deployment was likely
       deleted or never provisioned.
    3. **Consecutive reverse-proxy errors** — if the reverse proxy keeps
       returning HTTP 500 for ``_DEPLOY_HEALTH_MAX_PROXY_ERRORS`` probes,
       the pod is reachable but the training server inside never started.
    4. **Pod phase checking** — every few probes, queries the Basilica API to
       see if the pod has entered a terminal ``Failed`` phase.
    5. **Cross-deployment failure tracking** — if multiple consecutive
       deployments have all failed health checks, the Docker image is likely
       broken.  The error message escalates to tell the agent to stop
       recreating deployments.
    """
    global _deployment_failure_count

    deadline = time.time() + timeout
    attempts = 0
    last_detail = ""
    consecutive_server_errors = 0
    consecutive_dns_failures = 0
    consecutive_proxy_errors = 0

    while time.time() < deadline:
        attempts += 1
        healthy, detail = _probe_deployment_health(deployment_url, share_token)
        last_detail = detail
        if healthy:
            logger.info(
                "Deployment %s healthy after %d probes: %s",
                deployment_url, attempts, detail,
            )
            _deployment_failure_count = 0  # reset on success
            return json.dumps({
                "status": "ready",
                "url": deployment_url,
                "probes": attempts,
                "detail": detail,
            })

        # Track consecutive errors from the *actual server process* (tagged
        # ``[server]`` by ``_probe_deployment_health``).
        is_server_error = "[server]" in detail
        if is_server_error:
            consecutive_server_errors += 1
        else:
            consecutive_server_errors = 0

        # Track consecutive DNS / connection failures (hostname never resolves).
        # "Connection failed:" is the prefix from _probe_deployment_health for
        # URLError exceptions (DNS, refused, unreachable, etc.).
        is_dns_failure = detail.startswith("Connection failed:")
        if is_dns_failure:
            consecutive_dns_failures += 1
        else:
            consecutive_dns_failures = 0

        # Track consecutive reverse-proxy errors (pod reachable, server not).
        # ``[reverse-proxy]`` tagged errors mean the pod is running but the
        # training server process inside never started or crashed immediately.
        # A few of these are normal during pod startup, but if they persist
        # for many probes the server is never going to come up.
        is_proxy_error = "[reverse-proxy]" in detail
        if is_proxy_error:
            consecutive_proxy_errors += 1
        else:
            consecutive_proxy_errors = 0

        # Early abort: hostname never resolves — deployment may not exist
        if consecutive_dns_failures >= _DEPLOY_HEALTH_MAX_DNS_FAILURES:
            _deployment_failure_count += 1
            elapsed = int(time.time() - (deadline - timeout))
            logger.warning(
                "Deployment %s had %d consecutive DNS/connection failures "
                "over %ds — aborting health wait (last: %s) "
                "[%d consecutive deploy failures]",
                deployment_url, consecutive_dns_failures, elapsed, detail,
                _deployment_failure_count,
            )
            error_result: dict[str, Any] = {
                "status": "error",
                "error": (
                    f"Deployment hostname never resolved after "
                    f"{consecutive_dns_failures} consecutive probes ({elapsed}s)"
                ),
                "url": deployment_url,
                "last_probe": detail,
                "probes": attempts,
                "consecutive_deploy_failures": _deployment_failure_count,
                "hint": (
                    "The deployment URL hostname could not be resolved. "
                    "This usually means the deployment was deleted, never "
                    "finished provisioning, or the URL is incorrect. "
                    "Check get_training_deployment() for the deployment status, "
                    "then delete and recreate the deployment if needed."
                ),
            }
            if _deployment_failure_count >= _MAX_CONSECUTIVE_DEPLOY_FAILURES:
                error_result["hint"] = (
                    f"CRITICAL: {_deployment_failure_count} consecutive deployments "
                    f"have all failed. The deployment hostname never resolved, "
                    f"suggesting a persistent infrastructure issue. "
                    f"Do NOT create more deployments — report this as a blocker."
                )
                error_result["error_type"] = "environment"
                error_result["recoverable"] = False
            return json.dumps(error_result)

        # Early abort: reverse proxy keeps returning 500s — training server
        # inside the container never started (crash on import, OOM, etc.).
        if consecutive_proxy_errors >= _DEPLOY_HEALTH_MAX_PROXY_ERRORS:
            _deployment_failure_count += 1
            elapsed = int(time.time() - (deadline - timeout))
            logger.warning(
                "Deployment %s had %d consecutive reverse-proxy errors "
                "over %ds — training server never started (last: %s) "
                "[%d consecutive deploy failures]",
                deployment_url, consecutive_proxy_errors, elapsed, detail,
                _deployment_failure_count,
            )

            # Auto-fetch deployment logs for diagnostics.
            deploy_logs = ""
            try:
                from urllib.parse import urlparse
                host = urlparse(deployment_url).hostname or ""
                instance_name = host.split(".")[0] if "." in host else ""
                if instance_name:
                    client = _get_gpu_client()
                    deploy_logs = client.get_deployment_logs(
                        instance_name, tail=50,
                    )
            except Exception as log_exc:
                logger.debug("Could not fetch deploy logs: %s", log_exc)

            proxy_err: dict[str, Any] = {
                "status": "error",
                "error": (
                    f"Training server never started — reverse proxy returned "
                    f"HTTP 500 for {consecutive_proxy_errors} consecutive "
                    f"probes ({elapsed}s)"
                ),
                "url": deployment_url,
                "last_probe": detail,
                "probes": attempts,
                "consecutive_deploy_failures": _deployment_failure_count,
            }
            if deploy_logs and deploy_logs.strip():
                proxy_err["deploy_logs_tail"] = deploy_logs[-3000:]
            elif deploy_logs is not None:
                proxy_err["deploy_logs_tail"] = (
                    "(empty — container may not have started)"
                )
            if _deployment_failure_count >= _MAX_CONSECUTIVE_DEPLOY_FAILURES:
                proxy_err["hint"] = (
                    f"CRITICAL: {_deployment_failure_count} consecutive deployments "
                    f"have all failed to start the training server. "
                    f"The Docker image ({BASILICA_DEPLOY_IMAGE}) is likely broken "
                    f"or incompatible with the allocated GPU. "
                    f"Do NOT create more deployments — report this as a blocker."
                )
                proxy_err["error_type"] = "environment"
                proxy_err["recoverable"] = False
            else:
                proxy_err["hint"] = (
                    "The pod is running but the training server inside never "
                    "started. The reverse proxy can reach the pod but nothing "
                    "is listening on the expected port. Common causes: crash "
                    "during import (missing dependency, CUDA mismatch), OOM, or "
                    "the entrypoint script failed. Check deploy_logs_tail above "
                    "for the error, then delete and recreate the deployment."
                )
            return json.dumps(proxy_err)

        # Early abort: server is running but /health keeps crashing
        if consecutive_server_errors >= _DEPLOY_HEALTH_MAX_SERVER_ERRORS:
            _deployment_failure_count += 1
            logger.warning(
                "Deployment %s returned %d consecutive server errors — "
                "aborting health wait (last: %s) [%d consecutive deploy failures]",
                deployment_url, consecutive_server_errors, detail,
                _deployment_failure_count,
            )

            # Auto-fetch deployment logs so the agent gets diagnostics
            # without needing an extra tool call.
            deploy_logs = ""
            try:
                from urllib.parse import urlparse
                host = urlparse(deployment_url).hostname or ""
                instance_name = host.split(".")[0] if "." in host else ""
                if instance_name:
                    client = _get_gpu_client()
                    deploy_logs = client.get_deployment_logs(
                        instance_name, tail=50,
                    )
            except Exception as log_exc:
                logger.debug("Could not fetch deploy logs: %s", log_exc)

            error_result = {
                "status": "error",
                "error": (
                    f"Training server returned {consecutive_server_errors} "
                    f"consecutive HTTP 5xx errors"
                ),
                "url": deployment_url,
                "last_probe": detail,
                "probes": attempts,
                "consecutive_deploy_failures": _deployment_failure_count,
            }
            if deploy_logs and deploy_logs.strip():
                error_result["deploy_logs_tail"] = deploy_logs[-3000:]
            elif deploy_logs is not None:
                error_result["deploy_logs_tail"] = (
                    "(empty — container may not have started)"
                )
            if _deployment_failure_count >= _MAX_CONSECUTIVE_DEPLOY_FAILURES:
                error_result["hint"] = (
                    f"CRITICAL: {_deployment_failure_count} consecutive deployments "
                    f"have all failed health checks with the same error. "
                    f"The Docker image ({BASILICA_DEPLOY_IMAGE}) "
                    f"is likely broken — creating more deployments will not help. "
                    f"Report this as an infrastructure blocker in your finish output "
                    f"and do NOT create any more deployments."
                )
                error_result["error_type"] = "environment"
                error_result["recoverable"] = False
            else:
                error_result["hint"] = (
                    "The training server process is running but the /health "
                    "endpoint is failing. This usually means the server "
                    "crashed during startup or ResearchSession is broken. "
                    "Check deploy_logs_tail above for the traceback, then "
                    "delete and recreate the deployment."
                )
            return json.dumps(error_result)

        # Periodically check pod phase to detect Failed deployments early
        if attempts % _DEPLOY_HEALTH_PHASE_CHECK_INTERVAL == 0:
            try:
                client = _get_gpu_client()
                # Extract instance name from URL (UUID prefix of *.deployments.basilica.ai)
                from urllib.parse import urlparse
                host = urlparse(deployment_url).hostname or ""
                instance_name = host.split(".")[0] if "." in host else ""
                if instance_name:
                    resp = client.get_deployment(instance_name)
                    phase = getattr(resp, "phase", "")
                    # Log progress details for observability
                    raw_progress = getattr(resp, "progress", None)
                    if raw_progress is not None:
                        step = getattr(raw_progress, "current_step", "")
                        pct = getattr(raw_progress, "percentage", None)
                        elapsed = getattr(raw_progress, "elapsed_seconds", None)
                        pct_str = f" ({pct:.0f}%)" if pct is not None else ""
                        elapsed_str = f" [{elapsed}s]" if elapsed is not None else ""
                        logger.info(
                            "Deployment %s phase=%s step=%s%s%s",
                            instance_name, phase, step, pct_str, elapsed_str,
                        )
                    if phase and phase.lower() in ("failed", "error", "crashloopbackoff"):
                        _deployment_failure_count += 1
                        logger.warning(
                            "Deployment %s pod phase is %r — aborting health wait",
                            deployment_url, phase,
                        )
                        msg = getattr(resp, "message", "")
                        error_result: dict[str, Any] = {
                            "status": "error",
                            "error": f"Deployment pod phase is '{phase}'",
                            "url": deployment_url,
                            "phase": phase,
                            "last_probe": detail,
                            "probes": attempts,
                            "consecutive_deploy_failures": _deployment_failure_count,
                            "hint": (
                                "The deployment pod has entered a terminal failure "
                                "state. Check get_deployment_logs() for the error, "
                                "then delete and recreate the deployment."
                            ),
                        }
                        if msg:
                            error_result["message"] = msg
                        return json.dumps(error_result)
            except Exception as phase_exc:
                logger.debug("Phase check failed (non-fatal): %s", phase_exc)

        logger.debug(
            "Deployment %s not ready (probe %d): %s",
            deployment_url, attempts, detail,
        )
        time.sleep(_DEPLOY_HEALTH_INTERVAL)

    # Full timeout expired
    _deployment_failure_count += 1
    error_result = {
        "status": "error",
        "error": f"Deployment not ready after {timeout}s ({attempts} probes)",
        "url": deployment_url,
        "last_probe": last_detail,
        "consecutive_deploy_failures": _deployment_failure_count,
    }
    if _deployment_failure_count >= _MAX_CONSECUTIVE_DEPLOY_FAILURES:
        error_result["hint"] = (
            f"CRITICAL: {_deployment_failure_count} consecutive deployments have all "
            f"failed health checks. The Docker image is likely broken — creating "
            f"more deployments will not help. Report this as an infrastructure "
            f"blocker in your finish output and do NOT create more deployments."
        )
        error_result["error_type"] = "environment"
        error_result["recoverable"] = False
    else:
        error_result["hint"] = (
            "The deployment pod may be running but the training server inside "
            "has not started. Check get_deployment_logs() for startup errors."
        )
    return json.dumps(error_result)


@tool(
    description=(
        "Create a Basilica GPU deployment running the synth-city training server "
        "Docker image. The deployment starts a pre-built container with "
        "open-synth-miner already installed — no SSH or pip install needed. "
        "Returns the deployment URL for sending training requests. "
        "IMPORTANT: After creating a deployment, call wait_for_deployment_ready() "
        "with the returned URL — do NOT poll get_training_deployment() in a loop. "
        "wait_for_deployment_ready handles health probing, early abort on failures, "
        "and cross-deployment failure tracking automatically. "
        "name: deployment name (default: auto-generated). "
        "image: Docker image (default: from config). "
        "gpu_models: list of acceptable GPU models (e.g. ['A4000']). "
        "env: extra environment variables as JSON dict."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Deployment name (default: synth-city-trainer)",
            },
            "image": {
                "type": "string",
                "description": "Docker image to deploy (default: from config)",
            },
            "gpu_models": {
                "type": "string",
                "description": "JSON list of GPU model names, e.g. '[\"A4000\"]'",
            },
            "env": {
                "type": "string",
                "description": "Extra env vars as JSON dict",
            },
        },
        "required": [],
    },
)
def create_training_deployment(
    name: str = "",
    image: str = "",
    gpu_models: str = "",
    env: str = "",
) -> str:
    """Create a Basilica deployment with the synth-city GPU training image."""
    try:
        client = _get_gpu_client()
        deploy_name = name or _DEPLOYMENT_NAME_PREFIX
        deploy_image = image or BASILICA_DEPLOY_IMAGE
        # gpu_models may arrive as a JSON string or a native list from the LLM
        if isinstance(gpu_models, list):
            gpu_list = gpu_models
        else:
            gpu_list = json.loads(gpu_models) if gpu_models else None
        # env may arrive as a JSON string or a native dict from the LLM
        if isinstance(env, dict):
            env_dict = env
        else:
            env_dict = json.loads(env) if env else {}

        # Inject HF_TOKEN so the pod can download training data
        if HF_TOKEN and "HF_TOKEN" not in env_dict:
            env_dict["HF_TOKEN"] = HF_TOKEN

        resp = client.create_deployment(
            name=deploy_name,
            image=deploy_image,
            gpu_models=gpu_list,
            env=env_dict,
        )
        # Keep the consecutive-failure counter intact across create/delete
        # cycles — it is only reset when a deployment actually passes a health
        # check (inside wait_for_deployment_ready).  Resetting here would
        # prevent the CRITICAL threshold from ever being reached.
        # Extract progress info if available (e.g. scheduling, pulling)
        raw_progress = getattr(resp, "progress", None)
        progress = None
        if raw_progress is not None:
            progress = {
                "current_step": getattr(raw_progress, "current_step", None),
                "percentage": getattr(raw_progress, "percentage", None),
                "elapsed_seconds": getattr(raw_progress, "elapsed_seconds", None),
            }

        result: dict[str, Any] = {
            "status": "created",
            "instance_name": resp.instance_name,
            "url": resp.url,
            "phase": resp.phase,
            "message": getattr(resp, "message", None),
            "share_url": getattr(resp, "share_url", None),
            "share_token": getattr(resp, "share_token", None),
            "image": deploy_image,
            "next_step": (
                "Call wait_for_deployment_ready(deployment_url='"
                + (resp.url or "") + "') to wait for the training server "
                "to become healthy. Do NOT poll get_training_deployment."
            ),
        }
        if progress:
            result["progress"] = progress
        if _deployment_failure_count:
            result["consecutive_deploy_failures"] = _deployment_failure_count
            result["note"] = (
                f"WARNING: {_deployment_failure_count} previous consecutive "
                f"deployment(s) failed health checks. If this deployment also "
                f"fails, the Docker image may be broken."
            )
        return json.dumps(result, indent=2)
    except Exception as exc:
        err_str = str(exc).lower()
        is_infra = (
            "500" in err_str
            or "502" in err_str
            or "503" in err_str
            or "internal server error" in err_str
            or "connection" in err_str
        )
        result: dict[str, Any] = {
            "error": f"{type(exc).__name__}: {exc}",
        }
        if is_infra:
            result["error_type"] = "infrastructure"
            result["recoverable"] = True
            result["hint"] = (
                "Deployment creation failed due to a server-side error. "
                "This is likely a transient Basilica infrastructure issue. "
                "Wait a moment and try again."
            )
        return json.dumps(result)


@tool(
    description=(
        "Check the status of a Basilica training deployment. "
        "Returns phase (Pending/Running/Failed), URL, and pod info. "
        "NOTE: To wait for a deployment to become ready, use "
        "wait_for_deployment_ready() instead of polling this tool repeatedly — "
        "it handles health probing and early failure detection automatically."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Deployment instance name",
            },
        },
        "required": ["name"],
    },
)
def get_training_deployment(name: str) -> str:
    """Get deployment status including phase, progress, and pod details."""
    try:
        client = _get_gpu_client()
        resp = client.get_deployment(name)

        # Serialize replicas
        raw_replicas = getattr(resp, "replicas", None)
        replicas = None
        if raw_replicas is not None:
            if hasattr(raw_replicas, "__iter__"):
                replicas = [
                    {
                        "phase": getattr(r, "phase", None),
                        "ready": getattr(r, "ready", None),
                        "started": getattr(r, "started", None),
                        "name": getattr(r, "name", None),
                    }
                    for r in raw_replicas
                ]
            elif hasattr(raw_replicas, "desired"):
                # ReplicaStatus object with desired/ready counts
                replicas = {
                    "desired": getattr(raw_replicas, "desired", None),
                    "ready": getattr(raw_replicas, "ready", None),
                }
            else:
                replicas = str(raw_replicas)

        # Serialize progress (DeploymentProgress)
        raw_progress = getattr(resp, "progress", None)
        progress = None
        if raw_progress is not None:
            progress = {
                "current_step": getattr(raw_progress, "current_step", None),
                "percentage": getattr(raw_progress, "percentage", None),
                "elapsed_seconds": getattr(raw_progress, "elapsed_seconds", None),
            }

        # Serialize pods (list of PodInfo)
        raw_pods = getattr(resp, "pods", None)
        pods = None
        if raw_pods is not None and hasattr(raw_pods, "__iter__"):
            pods = [
                {
                    "name": getattr(p, "name", None),
                    "status": getattr(p, "status", None),
                    "node": getattr(p, "node", None),
                }
                for p in raw_pods
            ]

        result: dict[str, Any] = {
            "instance_name": resp.instance_name,
            "phase": resp.phase,
            "url": resp.url,
            "message": getattr(resp, "message", ""),
            "replicas": replicas,
            "created_at": str(getattr(resp, "created_at", "")),
        }

        # Only include progress/pods when present to keep output concise
        if progress:
            result["progress"] = progress
        if pods:
            result["pods"] = pods

        return json.dumps(result, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Get logs from a Basilica training deployment. "
        "Useful for debugging startup or training failures."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Deployment instance name",
            },
            "tail": {
                "type": "integer",
                "description": "Number of log lines to return (default: 100)",
            },
        },
        "required": ["name"],
    },
)
def get_deployment_logs(name: str, tail: int = 100) -> str:
    """Retrieve logs from a deployment."""
    try:
        client = _get_gpu_client()
        logs = client.get_deployment_logs(name, tail=tail)
        result: dict[str, Any] = {"instance_name": name, "logs": logs}
        if not logs or not logs.strip():
            result["hint"] = (
                "Logs are empty. The pod may still be starting (container "
                "pull / init in progress), or the training server crashed "
                "before producing any output. Try again in 30-60 seconds, "
                "or check the deployment phase with get_training_deployment()."
            )
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description="List all Basilica deployments for this account.",
)
def list_deployments() -> str:
    """List all deployments."""
    try:
        client = _get_gpu_client()
        deployments = client.list_deployments()
        return json.dumps({
            "deployments": deployments,
            "count": len(deployments),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Delete a Basilica training deployment and free GPU resources. "
        "Always call this when training is complete."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Deployment instance name",
            },
        },
        "required": ["name"],
    },
)
def delete_training_deployment(name: str) -> str:
    """Delete a deployment."""
    try:
        client = _get_gpu_client()
        result = client.delete_deployment(name)
        return json.dumps(result, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def _poll_job_result(
    deployment_url: str,
    job_id: str,
    share_token: str = "",
    max_polls: int = 40,
    poll_interval: float = 30.0,
) -> dict | None:
    """Poll GET /job-result/{job_id} to recover a result after stream drop.

    The training server caches completed results in memory.  If the NDJSON
    stream was severed by a reverse proxy before the final result line, the
    client can recover it here.

    Returns the result dict, or None if the job is unknown or still running
    after *max_polls* attempts.
    """
    import urllib.error
    import urllib.request

    url = deployment_url.rstrip("/") + f"/job-result/{job_id}"
    if share_token:
        url += f"?token={share_token}"

    logger.info(
        "Stream may have been severed — polling %s for job result "
        "(up to %d attempts, %ds interval)",
        url, max_polls, poll_interval,
    )

    for attempt in range(max_polls):
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read().decode())
                # Remove streaming metadata
                body.pop("type", None)
                body.pop("_pad", None)
                logger.info(
                    "Recovered job result for %s on poll %d/%d",
                    job_id, attempt + 1, max_polls,
                )
                return body
        except urllib.error.HTTPError as exc:
            if exc.code == 202:
                # Job still training — keep polling
                logger.debug(
                    "Job %s still training (poll %d/%d)",
                    job_id, attempt + 1, max_polls,
                )
            elif exc.code == 404:
                # Job unknown — server may be old (no /job-result endpoint)
                # or pod restarted (cache lost).
                logger.warning(
                    "Job %s not found on server (poll %d/%d) — "
                    "server may not support job recovery",
                    job_id, attempt + 1, max_polls,
                )
                return None
            else:
                logger.warning(
                    "Job result poll HTTP %d for %s (poll %d/%d)",
                    exc.code, job_id, attempt + 1, max_polls,
                )
        except Exception as exc:
            logger.warning(
                "Job result poll failed for %s (poll %d/%d): %s",
                job_id, attempt + 1, max_polls, exc,
            )

        if attempt < max_polls - 1:
            time.sleep(poll_interval)

    logger.warning("Job result recovery timed out for %s after %d polls", job_id, max_polls)
    return None


# ---------------------------------------------------------------------------
# Custom component code collection
# ---------------------------------------------------------------------------

_CUSTOM_COMPONENTS_DIR = os.path.join(os.getcwd(), "src", "models", "components")


def _collect_custom_component_code() -> str:
    """Read all .py files from src/models/components/ and concatenate them.

    Returns the combined source as a single string that the training server
    can write to disk and import.  Returns an empty string when the directory
    doesn't exist or contains no Python files.
    """
    if not os.path.isdir(_CUSTOM_COMPONENTS_DIR):
        return ""

    parts: list[str] = []
    for fname in sorted(os.listdir(_CUSTOM_COMPONENTS_DIR)):
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(_CUSTOM_COMPONENTS_DIR, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                parts.append(f"# --- {fname} ---\n{content}")
        except Exception as exc:
            logger.warning("Could not read component %s: %s", fpath, exc)

    combined = "\n\n".join(parts)
    if combined:
        logger.info(
            "Collected %d custom component file(s) (%d bytes) from %s",
            len(parts), len(combined), _CUSTOM_COMPONENTS_DIR,
        )
    return combined


@tool(
    description=(
        "Run an experiment on a Basilica training deployment via HTTP. "
        "The deployment must be running (created with create_training_deployment). "
        "Sends the experiment config to the training server and returns CRPS metrics. "
        "deployment_url: the URL from create_training_deployment. "
        "experiment: experiment config JSON from create_experiment. "
        "epochs: training epochs. "
        "timeframe: '5m' or '1m'."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "deployment_url": {
                "type": "string",
                "description": "Deployment URL (from create_training_deployment)",
            },
            "experiment": {
                "type": "string",
                "description": "Experiment config JSON",
            },
            "epochs": {
                "type": "integer",
                "description": "Training epochs (default: 1)",
            },
            "timeframe": {
                "type": "string",
                "enum": ["5m", "1m"],
                "description": "Data timeframe: '5m' (288-step) or '1m' (60-step)",
            },
            "share_token": {
                "type": "string",
                "description": (
                    "Share token for private deployments "
                    "(from create_training_deployment)"
                ),
            },
        },
        "required": ["deployment_url", "experiment"],
    },
)
def run_experiment_on_deployment(
    deployment_url: str,
    experiment: str,
    epochs: int = 1,
    timeframe: str = "5m",
    share_token: str = "",
) -> str:
    """Run a training experiment on a Basilica deployment via HTTP.

    Performs a lightweight health probe before sending the training request
    so that obviously-broken deployments fail fast instead of waiting through
    the full retry cycle.

    Retries automatically on HTTP 5xx errors with exponential backoff
    (30s, 60s, 120s, 240s) to ride out transient infrastructure issues.
    """
    import urllib.error
    import urllib.request

    # --- Pre-flight health check (with automatic wait) ---
    healthy, detail = _probe_deployment_health(deployment_url, share_token)
    if not healthy:
        logger.info(
            "Pre-flight health check failed for %s: %s — waiting for "
            "deployment to become ready before training",
            deployment_url, detail,
        )
        # Instead of failing immediately, wait for the deployment to become
        # healthy.  This handles the common case where the agent calls
        # run_experiment_on_deployment before the server has finished starting.
        wait_result_str = wait_for_deployment_ready(
            deployment_url,
            share_token=share_token,
            timeout=_DEPLOY_HEALTH_TIMEOUT,
        )
        wait_result = json.loads(wait_result_str)
        if wait_result.get("status") != "ready":
            logger.warning(
                "Deployment %s failed to become ready: %s",
                deployment_url, wait_result.get("error", "unknown"),
            )
            return wait_result_str

    try:
        exp_dict = json.loads(experiment) if isinstance(experiment, str) else experiment
        # Keep a copy with timeframe for provenance
        timeframe_tag = exp_dict.pop("timeframe", None) or timeframe
        _mon.emit("experiment", "experiment_start", name=f"deploy:{timeframe_tag}")

        tf_cfg = TIMEFRAME_CONFIGS.get(timeframe, TIMEFRAME_CONFIGS["5m"])
        asset_files = {
            hf_name: f"data/{hf_name}/{tf_cfg['file_suffix']}"
            for hf_name in SN50_TO_HF_ASSET.values()
        }

        import uuid as _uuid

        job_id = _uuid.uuid4().hex

        # Collect custom component source code (if any) to send to the
        # training server so it can load components not baked into the
        # Docker image.
        model_code_content = _collect_custom_component_code()

        if not model_code_content:
            logger.warning(
                "model_code_content is empty — no .py files found in %s. "
                "The training server requires model code to be provided.",
                _CUSTOM_COMPONENTS_DIR,
            )
            return json.dumps({
                "status": "error",
                "error": (
                    "No custom component code found. The training server "
                    "requires 'model_code_content' to be a non-empty string "
                    "containing the model code."
                ),
                "error_type": "configuration",
                "recoverable": True,
                "components_dir": _CUSTOM_COMPONENTS_DIR,
                "hint": (
                    f"The directory '{_CUSTOM_COMPONENTS_DIR}' is empty or "
                    f"does not exist. Use the ComponentAuthor agent or "
                    f"write_file tool to create at least one .py file with "
                    f"your model code (blocks/heads) in that directory, then "
                    f"retry run_experiment_on_deployment."
                ),
            })

        payload = {
            "experiment": exp_dict,
            "epochs": epochs,
            "timeframe": timeframe,
            "hf_repo": HF_TRAINING_DATA_REPO,
            "asset_files": asset_files,
            "input_len": int(tf_cfg["input_len"]),
            "pred_len": int(tf_cfg["pred_len"]),
            "job_id": job_id,
            "model_code_content": model_code_content,
        }

        # Build the /train URL
        url = deployment_url.rstrip("/")
        if share_token:
            url += f"/train?token={share_token}"
        else:
            url += "/train"

        body = json.dumps(payload).encode()
        logger.debug(
            "POST %s — payload keys: %s, experiment keys: %s, "
            "model_code_content length: %d, job_id: %s",
            url,
            list(payload.keys()),
            list(exp_dict.keys()) if isinstance(exp_dict, dict) else "N/A",
            len(model_code_content),
            job_id,
        )

        # --- Retry loop for transient HTTP 5xx errors ---
        last_http_error: urllib.error.HTTPError | None = None
        for attempt in range(_DEPLOY_MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=_DEPLOY_TRAIN_TIMEOUT) as resp:
                    result_text = resp.read().decode()

                # Success — break out of retry loop
                last_http_error = None
                break

            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    # Client error (4xx) — not retryable, log details and
                    # return a structured error directly instead of re-raising.
                    # (Re-raising loses the response body because exc.read()
                    # can only be consumed once.)
                    _err_body = ""
                    try:
                        _err_body = exc.read().decode()[:2000]
                    except Exception:
                        pass
                    logger.warning(
                        "Deployment returned HTTP %d (not retryable). "
                        "URL: %s | Response: %s | Payload keys: %s",
                        exc.code, url, _err_body, list(payload.keys()),
                    )
                    _mon.emit(
                        "system", "error",
                        message=f"Deployment HTTP {exc.code}: {exc.reason}",
                    )
                    client_err: dict[str, Any] = {
                        "status": "error",
                        "error": f"HTTP {exc.code}: {exc.reason}",
                        "response_body": _err_body,
                    }
                    if exc.code == 400:
                        client_err["hint"] = (
                            "The training server rejected the request payload. "
                            "Check response_body for the validation error. "
                            "Common causes: missing or empty 'model_code_content' "
                            "(ensure custom component .py files exist in "
                            f"'{_CUSTOM_COMPONENTS_DIR}'), or malformed "
                            "experiment config."
                        )
                    elif exc.code == 401:
                        client_err["hint"] = (
                            "The deployment returned 401 Unauthorized. "
                            "If the deployment was created with public=False, "
                            "you must pass the share_token from "
                            "create_training_deployment."
                        )
                    return json.dumps(client_err)
                # Server error (5xx) — retryable
                last_http_error = exc
                backoff = _DEPLOY_INITIAL_BACKOFF * (_DEPLOY_BACKOFF_FACTOR ** attempt)
                logger.warning(
                    "Deployment returned HTTP %d on attempt %d/%d — "
                    "retrying in %ds",
                    exc.code, attempt + 1, _DEPLOY_MAX_RETRIES, backoff,
                )
                _mon.emit(
                    "system", "deployment_retry",
                    attempt=attempt + 1,
                    http_code=exc.code,
                    backoff=backoff,
                )
                if attempt < _DEPLOY_MAX_RETRIES - 1:
                    time.sleep(backoff)

            except urllib.error.URLError as exc:
                # Connection-level failures (refused, reset, DNS) — also retryable
                backoff = _DEPLOY_INITIAL_BACKOFF * (_DEPLOY_BACKOFF_FACTOR ** attempt)
                logger.warning(
                    "Deployment connection failed on attempt %d/%d (%s) — "
                    "retrying in %ds",
                    attempt + 1, _DEPLOY_MAX_RETRIES, exc.reason, backoff,
                )
                if attempt < _DEPLOY_MAX_RETRIES - 1:
                    time.sleep(backoff)
                else:
                    _mon.emit(
                        "system", "error",
                        message=f"Deployment connection failed after "
                        f"{_DEPLOY_MAX_RETRIES} attempts: {exc.reason}",
                    )
                    return json.dumps({
                        "status": "error",
                        "error": f"Connection failed after {_DEPLOY_MAX_RETRIES} "
                        f"attempts: {exc.reason}",
                        "error_type": "infrastructure",
                        "recoverable": True,
                        "attempts": _DEPLOY_MAX_RETRIES,
                        "hint": (
                            "The deployment may not be ready or may have crashed. "
                            "Check get_training_deployment() for phase status and "
                            "get_deployment_logs() for errors. Consider deleting "
                            "and recreating the deployment."
                        ),
                    })

        # If we exhausted retries on 5xx errors
        if last_http_error is not None:
            error_body = ""
            try:
                error_body = last_http_error.read().decode()[:5000]
            except Exception:
                pass
            _mon.emit(
                "system", "error",
                message=f"Deployment HTTP {last_http_error.code} after "
                f"{_DEPLOY_MAX_RETRIES} retries",
            )
            return json.dumps({
                "status": "error",
                "error": f"HTTP {last_http_error.code}: {last_http_error.reason} "
                f"(after {_DEPLOY_MAX_RETRIES} retries with backoff)",
                "error_type": "infrastructure",
                "recoverable": True,
                "response_body": error_body,
                "attempts": _DEPLOY_MAX_RETRIES,
                "hint": (
                    "The deployment returned HTTP 5xx on all attempts. "
                    "The training server may be crashed or overloaded. "
                    "Try: 1) check get_deployment_logs() for errors, "
                    "2) delete and recreate the deployment, "
                    "3) if the problem persists, Basilica infrastructure "
                    "may be experiencing an outage."
                ),
            })

        # Parse the result — supports both NDJSON streams (new FastAPI server)
        # and plain JSON (legacy Flask server).
        #
        # NDJSON format: multiple JSON lines, each with a "type" field:
        #   {"type": "heartbeat", ...}   (keepalive, ignored)
        #   {"type": "result", ...}      (final training result)
        #   {"type": "error", ...}       (training failed)
        #
        # The client reads the *last* result/error line.  Heartbeat lines
        # are discarded.  If no typed lines are found, fall back to parsing
        # the entire response as a single JSON object (backward compat).
        result = None
        lines = result_text.strip().split("\n")
        if len(lines) > 1:
            # Multi-line → NDJSON stream
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    line_type = parsed.get("type", "")
                    if line_type in ("result", "error"):
                        result = parsed
                        # Remove streaming metadata before returning to agent
                        result.pop("type", None)
                        result.pop("_pad", None)
                except json.JSONDecodeError:
                    continue
        if result is None:
            # Single-line or no typed lines → plain JSON (backward compat)
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                pass

        # If we still have no result (stream severed before final line),
        # try to recover via the job-result endpoint.
        if result is None and job_id:
            result = _poll_job_result(deployment_url, job_id, share_token)

        if result is None:
            _mon.emit(
                "system", "error",
                message="Could not parse JSON from training server",
            )
            return json.dumps({
                "status": "error",
                "error": "Could not parse result from training server "
                "(stream may have been severed by reverse proxy)",
                "raw_output": result_text[:15000],
                "job_id": job_id,
                "hint": (
                    "The NDJSON stream was cut before the result line. "
                    "The server may still be training. Try polling "
                    f"GET /job-result/{job_id} on the deployment URL."
                ),
            })

        # --- Result tracking (mirrors run_experiment in research_tools) ---

        # Restore timeframe tag for provenance
        exp_dict["timeframe"] = timeframe_tag

        # Persist to Hippius (best-effort)
        try:
            from pipeline.tools.hippius_store import save_experiment_result
            save_experiment_result(f"deploy-{timeframe_tag}", exp_dict, result)
        except Exception:
            pass

        # Emit monitor event
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"
        _mon.emit(
            "experiment", "experiment_result",
            name=f"deploy:{timeframe_tag}",
            crps=metrics.get("crps"),
            status=status,
        )

        return json.dumps(result, indent=2, default=str)

    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode()[:5000]
        except Exception:
            pass
        _mon.emit("system", "error", message=f"Deployment HTTP {exc.code}: {exc.reason}")
        result_dict: dict[str, Any] = {
            "status": "error",
            "error": f"HTTP {exc.code}: {exc.reason}",
            "response_body": error_body,
        }
        if exc.code == 401:
            logger.warning(
                "Deployment returned 401 Unauthorized for URL %s. "
                "share_token provided: %s",
                deployment_url,
                bool(share_token),
            )
            result_dict["hint"] = (
                "The deployment returned 401 Unauthorized. "
                "If the deployment was created with public=False, you must pass "
                "the share_token from create_training_deployment. "
                "Alternatively, delete and recreate the deployment (it will now "
                "be created as public by default)."
            )
        return json.dumps(result_dict)
    except TimeoutError:
        _mon.emit("system", "error", message="Deployment training timed out")
        return json.dumps({
            "status": "error",
            "error": f"Training request timed out after {_DEPLOY_TRAIN_TIMEOUT}s.",
        })
    except Exception as exc:
        _mon.emit("system", "error", message=f"run_experiment_on_deployment: {exc}")
        return json.dumps({
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        })
