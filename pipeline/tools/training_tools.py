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

# Timeout for deployment health/readiness probes (seconds).
# Generous default: image pull (~60-120s) + CUDA/torch init (~30-60s) + Flask
# startup.  Total can easily reach 3-5 min on cold starts.
_DEPLOY_HEALTH_TIMEOUT = 600
_DEPLOY_HEALTH_INTERVAL = 15
# Abort early if the server returns this many consecutive HTTP 5xx responses
# with a non-empty body (meaning the Flask app responded but is broken —
# waiting longer won't help).  Empty-body 5xx from the reverse proxy are NOT
# counted here because they indicate the container is still starting.
_DEPLOY_HEALTH_MAX_SERVER_ERRORS = 5
# How often (in probes) to check the pod phase via the Basilica API
_DEPLOY_HEALTH_PHASE_CHECK_INTERVAL = 4
# Maximum time (seconds) to wait for the pod phase to become "Running"
# before starting health probes.  Uses a portion of the total timeout.
_DEPLOY_PHASE_WAIT_FRACTION = 0.5

_DEPLOYMENT_NAME_PREFIX = "synth-city-trainer"

# Training server port inside the container (must match Dockerfile EXPOSE)
_DEPLOY_SERVER_PORT = 8378


def _build_health_check():
    """Build a Kubernetes ``HealthCheckConfig`` for the training server.

    Configures readiness and startup probes so the reverse proxy only routes
    traffic once the Flask server is actually listening on ``/health``.
    Without these, the proxy returns empty-body HTTP 500 during container
    startup (image pull, CUDA init, torch import).

    Returns ``None`` if the SDK version doesn't support ``HealthCheckConfig``.
    """
    try:
        from basilica import HealthCheckConfig, ProbeConfig
    except ImportError:
        logger.debug("basilica SDK does not export HealthCheckConfig — skipping")
        return None

    return HealthCheckConfig(
        readiness=ProbeConfig(
            path="/health",
            port=_DEPLOY_SERVER_PORT,
            initial_delay_seconds=10,
            period_seconds=10,
            timeout_seconds=5,
            failure_threshold=3,
        ),
        startup=ProbeConfig(
            path="/health",
            port=_DEPLOY_SERVER_PORT,
            initial_delay_seconds=15,
            period_seconds=10,
            timeout_seconds=5,
            failure_threshold=30,  # 30 × 10s = 300s max startup time
        ),
    )

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
        except Exception as read_exc:
            logger.debug("Could not read HTTP error body: %s", read_exc)
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


def _extract_instance_name(deployment_url: str) -> str:
    """Extract the instance name (UUID) from a deployment URL."""
    from urllib.parse import urlparse
    host = urlparse(deployment_url).hostname or ""
    return host.split(".")[0] if "." in host else ""


def _fetch_deploy_logs_safe(instance_name: str, tail: int = 80) -> str:
    """Best-effort fetch of deployment logs; returns empty string on failure."""
    if not instance_name:
        return ""
    try:
        client = _get_gpu_client()
        logs = client.get_deployment_logs(instance_name, tail=tail)
        return (logs or "").strip()
    except Exception:
        return ""


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
    """Wait for a Basilica deployment to become healthy and ready for training.

    Uses a two-phase approach:

    **Phase 1 — Pod scheduling** (up to half the timeout):
    Polls the Basilica API until the pod phase becomes ``Running``.  No health
    probes are sent during this phase because the container hasn't started yet
    (image is being pulled, GPU is being allocated, etc.).

    **Phase 2 — Server health** (remaining timeout):
    Probes ``/health`` every 15 s until the training server responds HTTP 200.

    Early-abort mechanisms:
    1. **Terminal pod phase** — aborts immediately if the pod enters ``Failed``,
       ``Error``, or ``CrashLoopBackOff``.
    2. **Consecutive app errors** — aborts if the Flask app itself returns
       ``_DEPLOY_HEALTH_MAX_SERVER_ERRORS`` consecutive HTTP 5xx responses
       *with a non-empty body* (meaning the server is running but broken).
       Empty-body 5xx from the infrastructure proxy are NOT counted because
       they indicate the container is still starting.
    3. **Cross-deployment failure tracking** — escalates if multiple consecutive
       deployments fail.

    On any failure, deployment logs are automatically fetched and included in
    the error result so the agent doesn't need a separate ``get_deployment_logs``
    call.
    """
    global _deployment_failure_count

    instance_name = _extract_instance_name(deployment_url)
    deadline = time.time() + timeout
    phase_deadline = time.time() + timeout * _DEPLOY_PHASE_WAIT_FRACTION

    # ------------------------------------------------------------------
    # Phase 1: Wait for pod phase to become "Running" / "ready"
    # ------------------------------------------------------------------
    # Granular phases from the SDK (in lifecycle order):
    #   pending → scheduling → pulling → initializing → storage_sync
    #   → starting → health_check → ready
    # Terminal failure: failed
    # Terminal shutdown: terminating
    #
    # IMPORTANT: state="Active" does NOT mean the pod is serving traffic.
    # The SDK's own is_ready check requires:
    #   state in ("Active", "Running") AND replicas.ready > 0
    #   AND replicas.ready == replicas.desired
    # We mirror that logic here.
    _FAILED_PHASES = {"failed", "error", "crashloopbackoff"}

    # Pod-level statuses that indicate a terminal or semi-terminal problem.
    # These come from resp.pods[].status and are Kubernetes conditions.
    _POD_TERMINAL_STATUSES = {
        "imagepullbackoff",      # Image can't be pulled (bad tag, auth, etc.)
        "errimagepull",          # Same — immediate image pull failure
        "invalidimageformat",    # Corrupted or wrong-arch image
        "registryunavailable",   # Can't reach the registry
        "crashloopbackoff",      # Container keeps crashing on restart
        "oomkilled",             # Out of memory
        "createcontainererror",  # Bad container spec
    }

    def _is_deployment_ready(resp: Any) -> bool:
        """Check if a deployment is actually serving traffic.

        Mirrors the Basilica SDK's ``DeploymentStatus.is_ready`` logic:
        the high-level *state* must be "Active" or "Running" AND the
        replica counts must confirm at least one ready replica matching
        the desired count.  The granular *phase* of "ready" or "running"
        is also accepted as an authoritative signal.
        """
        phase = (getattr(resp, "phase", "") or "").lower()
        state = (getattr(resp, "state", "") or "").lower()

        # Granular phase "ready" or "running" is authoritative
        if phase in ("ready", "running"):
            return True

        # High-level state requires replica confirmation
        if state in ("active", "running"):
            replicas = getattr(resp, "replicas", None)
            if replicas is not None:
                desired = getattr(replicas, "desired", 0) or 0
                ready = getattr(replicas, "ready", 0) or 0
                return ready > 0 and ready >= desired

        return False

    def _check_pod_status(resp: Any) -> str | None:
        """Inspect per-pod status for terminal failures.

        Returns a human-readable error string if any pod is in a
        known-bad state (``ImagePullBackOff``, ``CrashLoopBackOff``,
        etc.), otherwise ``None``.
        """
        pods = getattr(resp, "pods", None) or []
        for pod in pods:
            status = (getattr(pod, "status", "") or "").lower()
            if status in _POD_TERMINAL_STATUSES:
                pod_name = getattr(pod, "name", "?")
                node = getattr(pod, "node", "?")
                return (
                    f"Pod {pod_name} is in terminal state '{status}' "
                    f"(node={node})"
                )
        return None

    def _summarize_deployment(resp: Any) -> str:
        """One-line summary of deployment for debug logging."""
        state = getattr(resp, "state", "?")
        phase = getattr(resp, "phase", "?")
        msg = getattr(resp, "message", "")
        replicas = getattr(resp, "replicas", None)
        r_desired = getattr(replicas, "desired", "?") if replicas else "?"
        r_ready = getattr(replicas, "ready", "?") if replicas else "?"
        pods = getattr(resp, "pods", None) or []
        pod_statuses = ", ".join(
            f"{getattr(p, 'name', '?')}={getattr(p, 'status', '?')}"
            for p in pods
        ) or "(no pods)"
        parts = [
            f"state={state}",
            f"phase={phase}",
            f"replicas={r_ready}/{r_desired}",
            f"pods=[{pod_statuses}]",
        ]
        if msg:
            parts.append(f"msg={msg}")
        return " ".join(parts)

    pod_phase = "unknown"
    pod_state = "unknown"
    phase_polls = 0
    while time.time() < min(phase_deadline, deadline):
        phase_polls += 1
        try:
            client = _get_gpu_client()
            if instance_name:
                resp = client.get_deployment(instance_name)
                pod_phase = (getattr(resp, "phase", "") or "").lower()
                pod_state = (getattr(resp, "state", "") or "").lower()

                if _is_deployment_ready(resp):
                    logger.info(
                        "Deployment %s is ready (%s) after %d polls",
                        deployment_url,
                        _summarize_deployment(resp),
                        phase_polls,
                    )
                    break

                if pod_phase in _FAILED_PHASES or pod_state in _FAILED_PHASES:
                    _deployment_failure_count += 1
                    logs = _fetch_deploy_logs_safe(instance_name)
                    summary = _summarize_deployment(resp)
                    logger.warning(
                        "Deployment %s failed: %s",
                        deployment_url, summary,
                    )
                    return json.dumps(
                        _build_wait_error(
                            deployment_url=deployment_url,
                            error=f"Deployment failed: {summary}",
                            last_probe=summary,
                            probes=0,
                            deploy_logs=logs,
                            phase=pod_phase,
                        )
                    )

                # Check individual pod statuses for terminal failures
                # (e.g. ImagePullBackOff, CrashLoopBackOff) that the
                # high-level state/phase may not reflect yet.
                pod_error = _check_pod_status(resp)
                if pod_error:
                    _deployment_failure_count += 1
                    logs = _fetch_deploy_logs_safe(instance_name)
                    summary = _summarize_deployment(resp)
                    logger.warning(
                        "Deployment %s has pod-level failure: %s (%s)",
                        deployment_url, pod_error, summary,
                    )
                    return json.dumps(
                        _build_wait_error(
                            deployment_url=deployment_url,
                            error=pod_error,
                            last_probe=summary,
                            probes=0,
                            deploy_logs=logs,
                            phase=pod_phase,
                        )
                    )

                # Detect GPU scheduling failure: if the deployment is
                # Active but replicas.desired is still 0 after several
                # polls, no GPU nodes match the request.  Waiting longer
                # won't help — the agent should try different GPU models.
                replicas = getattr(resp, "replicas", None)
                r_desired = getattr(replicas, "desired", None) if replicas else None
                if (
                    phase_polls >= 6  # ~90s of polling
                    and pod_state == "active"
                    and r_desired is not None
                    and int(r_desired) == 0
                ):
                    _deployment_failure_count += 1
                    summary = _summarize_deployment(resp)
                    logger.warning(
                        "Deployment %s stuck with replicas=0/0 — "
                        "no GPU nodes match the request: %s",
                        deployment_url, summary,
                    )
                    return json.dumps(
                        _build_wait_error(
                            deployment_url=deployment_url,
                            error=(
                                "No GPU nodes available: deployment is Active but "
                                "desired replicas = 0 (no pods scheduled). "
                                "The Basilica cluster has no GPU nodes matching "
                                "your requested gpu_models. Try omitting gpu_models "
                                "to accept any available GPU, or try again later."
                            ),
                            last_probe=summary,
                            probes=0,
                            deploy_logs="",
                            phase=pod_phase,
                        )
                    )

        except Exception as exc:
            logger.debug("Phase poll %d failed (non-fatal): %s", phase_polls, exc)

        logger.debug(
            "Deployment %s phase=%s state=%s (poll %d), waiting...",
            deployment_url, pod_phase, pod_state, phase_polls,
        )
        time.sleep(_DEPLOY_HEALTH_INTERVAL)
    else:
        # Phase 1 exhausted its budget — pod never reached Running.
        # Fall through to Phase 2 anyway; the health probes will either
        # succeed (if the pod started late) or time out.
        logger.info(
            "Deployment %s still state=%s phase=%s after %d polls "
            "(%.0fs) — proceeding to health probes",
            deployment_url, pod_state, pod_phase, phase_polls,
            timeout * _DEPLOY_PHASE_WAIT_FRACTION,
        )

    # ------------------------------------------------------------------
    # Phase 2: Probe /health until the server is ready
    # ------------------------------------------------------------------
    health_probes = 0
    last_detail = ""
    consecutive_server_errors = 0

    while time.time() < deadline:
        health_probes += 1
        healthy, detail = _probe_deployment_health(deployment_url, share_token)
        last_detail = detail

        if healthy:
            logger.info(
                "Deployment %s healthy after %d health probes: %s",
                deployment_url, health_probes, detail,
            )
            _deployment_failure_count = 0
            return json.dumps({
                "status": "ready",
                "url": deployment_url,
                "probes": health_probes,
                "detail": detail,
            })

        # Track consecutive errors from the *actual server process* (tagged
        # ``[server]`` by ``_probe_deployment_health``).  Reverse-proxy errors
        # (``[reverse-proxy]``) mean the container isn't ready yet — those are
        # transient and should be waited through, not trigger early abort.
        is_server_error = "[server]" in detail
        if is_server_error:
            consecutive_server_errors += 1
        else:
            consecutive_server_errors = 0

        # Early abort: Flask app is running but /health keeps crashing
        if consecutive_server_errors >= _DEPLOY_HEALTH_MAX_SERVER_ERRORS:
            _deployment_failure_count += 1
            logs = _fetch_deploy_logs_safe(instance_name)
            logger.warning(
                "Deployment %s returned %d consecutive app server errors — "
                "aborting health wait (last: %s) [%d consecutive deploy failures]",
                deployment_url, consecutive_server_errors, detail,
                _deployment_failure_count,
            )
            return json.dumps(
                _build_wait_error(
                    deployment_url=deployment_url,
                    error=(
                        f"Training server returned {consecutive_server_errors} "
                        f"consecutive HTTP 5xx errors"
                    ),
                    last_probe=detail,
                    probes=health_probes,
                    deploy_logs=logs,
                )
            )

        # Periodically check pod phase + pod status to detect crashes
        if health_probes % _DEPLOY_HEALTH_PHASE_CHECK_INTERVAL == 0:
            try:
                client = _get_gpu_client()
                if instance_name:
                    resp = client.get_deployment(instance_name)
                    phase = (getattr(resp, "phase", "") or "").lower()
                    state = (getattr(resp, "state", "") or "").lower()

                    # Check deployment-level failure
                    if phase in _FAILED_PHASES or state in _FAILED_PHASES:
                        _deployment_failure_count += 1
                        logs = _fetch_deploy_logs_safe(instance_name)
                        summary = _summarize_deployment(resp)
                        logger.warning(
                            "Deployment %s failed during health wait: %s",
                            deployment_url, summary,
                        )
                        return json.dumps(
                            _build_wait_error(
                                deployment_url=deployment_url,
                                error=f"Deployment failed: {summary}",
                                last_probe=detail,
                                probes=health_probes,
                                deploy_logs=logs,
                                phase=phase,
                            )
                        )

                    # Check pod-level terminal failures
                    pod_error = _check_pod_status(resp)
                    if pod_error:
                        _deployment_failure_count += 1
                        logs = _fetch_deploy_logs_safe(instance_name)
                        summary = _summarize_deployment(resp)
                        logger.warning(
                            "Deployment %s pod failure during health "
                            "wait: %s (%s)",
                            deployment_url, pod_error, summary,
                        )
                        return json.dumps(
                            _build_wait_error(
                                deployment_url=deployment_url,
                                error=pod_error,
                                last_probe=detail,
                                probes=health_probes,
                                deploy_logs=logs,
                                phase=phase,
                            )
                        )

                    logger.debug(
                        "Phase check during health wait: %s",
                        _summarize_deployment(resp),
                    )
            except Exception as phase_exc:
                logger.debug("Phase check failed (non-fatal): %s", phase_exc)

        logger.debug(
            "Deployment %s not ready (probe %d): %s",
            deployment_url, health_probes, detail,
        )
        time.sleep(_DEPLOY_HEALTH_INTERVAL)

    # Full timeout expired — fetch final deployment status for diagnostics
    _deployment_failure_count += 1
    logs = _fetch_deploy_logs_safe(instance_name)
    final_summary = ""
    if instance_name:
        try:
            resp = _get_gpu_client().get_deployment(instance_name)
            final_summary = _summarize_deployment(resp)
        except Exception:
            final_summary = "(could not fetch final status)"
    return json.dumps(
        _build_wait_error(
            deployment_url=deployment_url,
            error=(
                f"Deployment not ready after {timeout}s "
                f"({health_probes} health probes). "
                f"Final status: {final_summary}"
            ),
            last_probe=last_detail,
            probes=health_probes,
            deploy_logs=logs,
        )
    )


def _build_wait_error(
    *,
    deployment_url: str,
    error: str,
    last_probe: str,
    probes: int,
    deploy_logs: str,
    phase: str = "",
) -> dict[str, Any]:
    """Build a structured error dict for ``wait_for_deployment_ready`` failures.

    Includes deployment logs (when available) so the agent can diagnose the
    problem without a separate ``get_deployment_logs()`` call.
    """
    result: dict[str, Any] = {
        "status": "error",
        "error": error,
        "url": deployment_url,
        "last_probe": last_probe,
        "probes": probes,
        "consecutive_deploy_failures": _deployment_failure_count,
    }
    if phase:
        result["phase"] = phase
    result["deploy_logs_tail"] = deploy_logs or "(empty — container may not have started)"

    if _deployment_failure_count >= _MAX_CONSECUTIVE_DEPLOY_FAILURES:
        result["hint"] = (
            f"CRITICAL: {_deployment_failure_count} consecutive deployments "
            f"have all failed health checks. "
            f"The Docker image (ghcr.io/tensorlink-ai/synth-city-gpu:latest) "
            f"is likely broken — creating more deployments will not help. "
            f"Report this as an infrastructure blocker in your finish output "
            f"and do NOT create any more deployments."
        )
        result["error_type"] = "environment"
        result["recoverable"] = False
    elif deploy_logs:
        result["hint"] = (
            "The training server process started but is unhealthy. "
            "Check the deploy_logs_tail above for the error, then "
            "delete and recreate the deployment."
        )
    else:
        result["hint"] = (
            "The pod is running but the training server inside never started "
            "and produced no logs. This usually means the container image "
            "could not be pulled (wrong tag, GHCR auth, deleted image) or "
            "the container crashed immediately (OOM before Python started, "
            "CUDA driver mismatch, segfault in torch import). "
            "Check the deployment status fields (state, phase, pods) "
            "for the specific Kubernetes failure reason. "
            "If the image cannot be pulled, this is an infrastructure "
            "issue — creating more deployments will not help."
        )
    return result


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
    global _deployment_failure_count
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

        # Build Kubernetes health probes so the reverse proxy only routes
        # traffic once the training server is actually listening.  Without
        # this, the proxy returns empty-body HTTP 500 during container startup.
        health_check = _build_health_check()

        resp = client.create_deployment(
            name=deploy_name,
            image=deploy_image,
            gpu_models=gpu_list,
            env=env_dict,
            health_check=health_check,
        )
        # Reset the consecutive-failure counter on successful deployment
        # creation so a previous string of failures doesn't permanently
        # block new attempts after the underlying issue is fixed.
        prev_failures = _deployment_failure_count
        _deployment_failure_count = 0
        result: dict[str, Any] = {
            "status": "created",
            "instance_name": resp.instance_name,
            "url": resp.url,
            "phase": resp.phase,
            "share_url": getattr(resp, "share_url", None),
            "share_token": getattr(resp, "share_token", None),
            "image": deploy_image,
        }
        if prev_failures:
            result["note"] = (
                f"Reset consecutive deployment failure counter "
                f"(was {prev_failures})."
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

        # replicas is a ReplicaStatus(desired, ready) — NOT a list
        raw_replicas = getattr(resp, "replicas", None)
        replicas = None
        if raw_replicas is not None:
            replicas = {
                "desired": getattr(raw_replicas, "desired", None),
                "ready": getattr(raw_replicas, "ready", None),
            }

        # Per-pod detail (PodInfo objects with name, status, node)
        raw_pods = getattr(resp, "pods", None) or []
        pods = [
            {
                "name": getattr(p, "name", None),
                "status": getattr(p, "status", None),
                "node": getattr(p, "node", None),
            }
            for p in raw_pods
        ] if hasattr(raw_pods, "__iter__") else None

        # Progress info (current_step, percentage, elapsed_seconds)
        raw_progress = getattr(resp, "progress", None)
        progress = None
        if raw_progress is not None:
            progress = {
                "current_step": getattr(raw_progress, "current_step", None),
                "percentage": getattr(raw_progress, "percentage", None),
                "elapsed_seconds": getattr(raw_progress, "elapsed_seconds", None),
            }

        return json.dumps({
            "instance_name": resp.instance_name,
            "state": getattr(resp, "state", None),
            "phase": resp.phase,
            "url": resp.url,
            "message": getattr(resp, "message", ""),
            "replicas": replicas,
            "pods": pods if pods else None,
            "progress": progress,
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
                    # Client error (4xx) — not retryable, log details
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
