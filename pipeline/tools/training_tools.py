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
            f"tried = ['src.research.agent_api', 'research.agent_api']\n"
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

_DEPLOYMENT_NAME_PREFIX = "synth-city-trainer"


def _probe_deployment_health(url: str, share_token: str = "") -> tuple[bool, str]:
    """Send a lightweight GET probe to a deployment URL.

    Returns ``(healthy, detail)`` where *healthy* is True if the server
    responded with a 2xx status.  The probe hits the root path ``/`` — most
    HTTP servers return *something* even without an explicit health endpoint.
    """
    import urllib.error
    import urllib.request

    probe_url = url.rstrip("/") + "/"
    if share_token:
        probe_url += f"?token={share_token}"
    try:
        req = urllib.request.Request(probe_url, method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return True, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        # 4xx means the server is up but the path is wrong — still healthy
        if 400 <= exc.code < 500:
            return True, f"HTTP {exc.code} (server responsive)"
        return False, f"HTTP {exc.code}: {exc.reason}"
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
    """Probe a deployment URL until the HTTP server is responsive."""
    deadline = time.time() + timeout
    attempts = 0
    last_detail = ""

    while time.time() < deadline:
        attempts += 1
        healthy, detail = _probe_deployment_health(deployment_url, share_token)
        last_detail = detail
        if healthy:
            logger.info(
                "Deployment %s healthy after %d probes: %s",
                deployment_url, attempts, detail,
            )
            return json.dumps({
                "status": "ready",
                "url": deployment_url,
                "probes": attempts,
                "detail": detail,
            })
        logger.debug(
            "Deployment %s not ready (probe %d): %s",
            deployment_url, attempts, detail,
        )
        time.sleep(_DEPLOY_HEALTH_INTERVAL)

    return json.dumps({
        "status": "error",
        "error": f"Deployment not ready after {timeout}s ({attempts} probes)",
        "url": deployment_url,
        "last_probe": last_detail,
        "hint": (
            "The deployment pod may be running but the training server inside "
            "has not started. Check get_deployment_logs() for startup errors."
        ),
    })


@tool(
    description=(
        "Create a Basilica GPU deployment running the synth-city training server "
        "Docker image. The deployment starts a pre-built container with "
        "open-synth-miner already installed — no SSH or pip install needed. "
        "Returns the deployment URL for sending training requests. "
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
        return json.dumps({
            "status": "created",
            "instance_name": resp.instance_name,
            "url": resp.url,
            "phase": resp.phase,
            "share_url": getattr(resp, "share_url", None),
            "share_token": getattr(resp, "share_token", None),
            "image": deploy_image,
        }, indent=2)
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
        "Returns phase (Pending/Running/Failed), URL, and pod info."
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
    """Get deployment status."""
    try:
        client = _get_gpu_client()
        resp = client.get_deployment(name)
        # Serialize replicas (list of SDK ReplicaStatus objects) to plain dicts
        raw_replicas = getattr(resp, "replicas", None)
        replicas = None
        if raw_replicas is not None:
            replicas = [
                {
                    "phase": getattr(r, "phase", None),
                    "ready": getattr(r, "ready", None),
                    "started": getattr(r, "started", None),
                    "name": getattr(r, "name", None),
                }
                for r in raw_replicas
            ] if hasattr(raw_replicas, "__iter__") else str(raw_replicas)
        return json.dumps({
            "instance_name": resp.instance_name,
            "phase": resp.phase,
            "url": resp.url,
            "message": getattr(resp, "message", ""),
            "replicas": replicas,
            "created_at": str(getattr(resp, "created_at", "")),
        }, indent=2)
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
        return json.dumps({"instance_name": name, "logs": logs})
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

    Retries automatically on HTTP 5xx errors with exponential backoff
    (30s, 60s, 120s, 240s) to ride out transient infrastructure issues.
    """
    import urllib.error
    import urllib.request

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

        payload = {
            "experiment": exp_dict,
            "epochs": epochs,
            "timeframe": timeframe,
            "hf_repo": HF_TRAINING_DATA_REPO,
            "asset_files": asset_files,
            "input_len": int(tf_cfg["input_len"]),
            "pred_len": int(tf_cfg["pred_len"]),
        }

        # Build the /train URL
        url = deployment_url.rstrip("/")
        if share_token:
            url += f"/train?token={share_token}"
        else:
            url += "/train"

        body = json.dumps(payload).encode()

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
                    # Client error (4xx) — not retryable
                    raise
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

        # Parse the result
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            _mon.emit(
                "system", "error",
                message="Could not parse JSON from training server",
            )
            return json.dumps({
                "status": "error",
                "error": "Could not parse JSON from training server",
                "raw_output": result_text[:15000],
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
