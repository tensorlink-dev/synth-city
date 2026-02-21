"""
synth-city HTTP bridge server.

Exposes synth-city's pipeline operations and research tools as a lightweight
HTTP API.  Works standalone (curl, Python, any HTTP client) or as a backend
for agent frameworks like OpenClaw.

Start with::

    python main.py bridge            # default: 127.0.0.1:8377
    python main.py bridge --port 9000

Then interact via curl::

    curl http://127.0.0.1:8377/health
    curl http://127.0.0.1:8377/components/blocks
    curl -X POST http://127.0.0.1:8377/pipeline/run -d '{}'

Or the built-in CLI client::

    python main.py client blocks
    python main.py client status
    python main.py client price BTC

Endpoints
---------
GET  /health                  → liveness check
POST /pipeline/run            → kick off a full pipeline run (async)
GET  /pipeline/status         → poll current pipeline status
POST /experiment/create       → create an experiment config
POST /experiment/run          → run an experiment and return metrics
POST /experiment/validate     → validate an experiment config
GET  /experiment/compare      → compare all session results
GET  /components/blocks       → list available backbone blocks
GET  /components/heads        → list available head types
GET  /components/presets      → list ready-to-run presets
GET  /session/summary         → current research session summary
POST /session/clear           → reset research session
GET  /market/price/:asset     → live price from Pyth oracle
GET  /market/history/:asset   → historical OHLCV data
GET  /registry/files          → list component files in the registry
GET  /registry/read           → read a component source file
POST /registry/write          → write a new block/head component
POST /registry/reload         → reload the component registry
GET  /hf/models               → list models on HF Hub
GET  /hf/model-card           → fetch model card from HF Hub
GET  /hf/artifact             → download a JSON artifact from HF Hub
GET  /history/runs            → list pipeline runs from Hippius
GET  /history/run/:run_id     → load a specific run from Hippius
GET  /history/experiments     → best experiments across all runs
GET  /history/trackio         → fetch experiment runs from Hippius
GET  /agents/list             → list agent modules
GET  /agents/read             → read an agent source file
POST /agents/write            → write a new agent module
GET  /agents/prompts/list     → list prompt modules
GET  /agents/prompts/read     → read a prompt source file
POST /agents/prompts/write    → write a new prompt module
GET  /agents/tools            → list all registered tool names
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

# Maximum request body size (1 MB) to prevent memory exhaustion
MAX_CONTENT_LENGTH = 1 * 1024 * 1024

# Optional API key — when set, every request must include a matching
# ``X-API-Key`` header.  When empty (the default), authentication is
# disabled so localhost use works without configuration.
BRIDGE_API_KEY: str = os.getenv("BRIDGE_API_KEY", "")

# Valid SN50 asset identifiers
VALID_ASSETS = frozenset({
    "BTC", "ETH", "SOL", "XAU", "SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX",
})

# Alphanumeric + underscore pattern for asset path segments
_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9_]+$")


# ---------------------------------------------------------------------------
# Pipeline state (shared across requests)
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """Tracks the status of asynchronous pipeline runs."""

    running: bool = False
    status: str = "idle"
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    current_stage: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "status": self.status,
            "current_stage": self.current_stage,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": (
                round(time.time() - self.started_at, 1)
                if self.started_at and self.running
                else None
            ),
            "result": self.result,
            "error": self.error,
        }

    def reset(self) -> None:
        """Clear state for a new pipeline run."""
        self.running = True
        self.status = "running"
        self.started_at = time.time()
        self.finished_at = None
        self.result = None
        self.error = None
        self.current_stage = "initializing"

    def mark_completed(self, result: dict[str, Any]) -> None:
        self.running = False
        self.status = "completed"
        self.finished_at = time.time()
        self.result = result
        self.current_stage = ""

    def mark_failed(self, exc: Exception) -> None:
        self.running = False
        self.status = "failed"
        self.finished_at = time.time()
        self.error = f"{type(exc).__name__}: {exc}"
        self.current_stage = ""


_pipeline_state = PipelineState()
_pipeline_lock = threading.Lock()


def _run_pipeline_async(task: dict[str, Any]) -> None:
    """Execute the pipeline in a background thread."""
    try:
        from pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(
            max_retries=task.get("retries", 5),
            base_temperature=task.get("temperature", 0.1),
            publish=task.get("publish", False),
        )
        result = orchestrator.run(task)

        with _pipeline_lock:
            _pipeline_state.mark_completed(result)

    except Exception as exc:
        logger.exception("Pipeline failed")
        with _pipeline_lock:
            _pipeline_state.mark_failed(exc)


# ---------------------------------------------------------------------------
# Research tool wrappers (lazy-loaded to avoid import overhead)
# ---------------------------------------------------------------------------

_tools_loaded = False


def _ensure_tools_loaded() -> None:
    """Import tool modules so they register with the tool registry."""
    global _tools_loaded
    if _tools_loaded:
        return
    try:
        import pipeline.tools.agent_tools  # noqa: F401
        import pipeline.tools.analysis_tools  # noqa: F401
        import pipeline.tools.hippius_store  # noqa: F401
        import pipeline.tools.market_data  # noqa: F401
        import pipeline.tools.register_tools  # noqa: F401
        import pipeline.tools.research_tools  # noqa: F401
        _tools_loaded = True
    except Exception:
        logger.exception("Failed to load tool modules")
        raise


def _research_call(tool_name: str, **kwargs: Any) -> dict[str, Any]:
    """Call a registered synth-city tool by name and return the parsed result."""
    from pipeline.tools.registry import get_tool

    td = get_tool(tool_name)
    if td is None:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        raw = td.func(**kwargs)
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"result": raw}
        return raw if isinstance(raw, dict) else {"result": raw}
    except Exception as exc:
        logger.exception("Tool %s failed", tool_name)
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_asset(asset: str) -> str | None:
    """Return an error string if *asset* is invalid, else ``None``."""
    if not asset or not _SAFE_PATH_RE.match(asset):
        return f"Invalid asset identifier: {asset!r}"
    asset_upper = asset.upper()
    if asset_upper not in VALID_ASSETS:
        return f"Unknown asset: {asset!r}. Valid assets: {', '.join(sorted(VALID_ASSETS))}"
    return None


def _validate_experiment_body(body: dict[str, Any]) -> str | None:
    """Return an error string if required experiment fields are missing."""
    if "blocks" not in body and "experiment" not in body:
        return "Missing required field: 'blocks' or 'experiment'"
    return None


def _validate_positive_int(value: Any, name: str) -> tuple[int | None, str | None]:
    """Coerce *value* to a positive int. Returns (value, error)."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None, f"{name} must be an integer, got {type(value).__name__}"
    if v <= 0:
        return None, f"{name} must be positive, got {v}"
    return v, None


def _validate_positive_float(value: Any, name: str) -> tuple[float | None, str | None]:
    """Coerce *value* to a positive float. Returns (value, error)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None, f"{name} must be a number, got {type(value).__name__}"
    if v <= 0:
        return None, f"{name} must be positive, got {v}"
    return v, None


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class BridgeHandler(BaseHTTPRequestHandler):
    """Routes HTTP requests to synth-city operations."""

    def log_message(self, format: str, *args: Any) -> None:
        logger.info(format, *args)

    # ---- auth
    def _check_auth(self) -> bool:
        """Validate the ``X-API-Key`` header if ``BRIDGE_API_KEY`` is set.

        Returns ``True`` when the request is authorised (or auth is disabled).
        Sends a 401 response and returns ``False`` otherwise.
        """
        if not BRIDGE_API_KEY:
            return True
        provided = self.headers.get("X-API-Key", "")
        if provided == BRIDGE_API_KEY:
            return True
        self._send_json({"error": "Invalid or missing API key"}, status=401)
        return False

    # ---- helpers
    def _send_json(self, data: Any, status: int = 200) -> None:
        try:
            body = json.dumps(data, indent=2, default=str).encode()
        except (TypeError, ValueError) as exc:
            body = json.dumps({"error": f"Serialization error: {exc}"}).encode()
            status = 500
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, Any] | None:
        """Read and parse the JSON request body.

        Returns ``None`` (and sends a 400 response) on parse errors or
        oversized payloads.
        """
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError:
            self._send_json(
                {"error": f"Invalid Content-Length: {raw_length!r}"}, status=400,
            )
            return None

        if length > MAX_CONTENT_LENGTH:
            self._send_json(
                {"error": f"Request body too large ({length} bytes, max {MAX_CONTENT_LENGTH})"},
                status=413,
            )
            return None

        if length == 0:
            return {}

        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid JSON: {exc}"}, status=400)
            return None

        if not isinstance(parsed, dict):
            self._send_json(
                {"error": f"Expected JSON object, got {type(parsed).__name__}"}, status=400,
            )
            return None

        return parsed

    def _path_parts(self) -> tuple[str, dict[str, list[str]]]:
        parsed = urlparse(self.path)
        return parsed.path.rstrip("/"), parse_qs(parsed.query)

    def _load_tools_or_fail(self) -> bool:
        """Load tool modules. Returns ``True`` on success, sends 500 on failure."""
        try:
            _ensure_tools_loaded()
            return True
        except Exception as exc:
            self._send_json(
                {"error": f"Failed to load tools: {exc}"}, status=500,
            )
            return False

    # ---- GET routes
    def do_GET(self) -> None:
        if not self._check_auth():
            return
        path, qs = self._path_parts()

        if path == "/health":
            self._send_json({"status": "ok", "service": "synth-city-bridge"})
            return

        if path == "/pipeline/status":
            with _pipeline_lock:
                self._send_json(_pipeline_state.to_dict())
            return

        if path == "/components/blocks":
            if self._load_tools_or_fail():
                self._send_json(_research_call("list_blocks"))

        elif path == "/components/heads":
            if self._load_tools_or_fail():
                self._send_json(_research_call("list_heads"))

        elif path == "/components/presets":
            if self._load_tools_or_fail():
                self._send_json(_research_call("list_presets"))

        elif path == "/experiment/compare":
            if self._load_tools_or_fail():
                self._send_json(_research_call("compare_results"))

        elif path == "/session/summary":
            if self._load_tools_or_fail():
                self._send_json(_research_call("session_summary"))

        elif path.startswith("/market/price/"):
            asset = path.split("/")[-1]
            err = _validate_asset(asset)
            if err:
                self._send_json({"error": err}, status=400)
                return
            if self._load_tools_or_fail():
                self._send_json(_research_call("get_latest_price", asset=asset.upper()))

        elif path.startswith("/market/history/"):
            asset = path.split("/")[-1]
            err = _validate_asset(asset)
            if err:
                self._send_json({"error": err}, status=400)
                return
            raw_days = qs.get("days", ["30"])[0]
            days, days_err = _validate_positive_int(raw_days, "days")
            if days_err:
                self._send_json({"error": days_err}, status=400)
                return
            if self._load_tools_or_fail():
                self._send_json(
                    _research_call("get_historical_data", asset=asset.upper(), days=days),
                )

        # ---- registry / component discovery
        elif path == "/registry/files":
            if self._load_tools_or_fail():
                self._send_json(_research_call("list_component_files"))

        elif path == "/registry/read":
            file_path = qs.get("path", [""])[0]
            if not file_path:
                self._send_json(
                    {"error": "Missing required query parameter: 'path'"}, status=400,
                )
                return
            if self._load_tools_or_fail():
                result = _research_call("read_component", path=file_path)
                self._send_json(result)

        # ---- HF Hub
        elif path == "/hf/models":
            repo_id = qs.get("repo_id", [""])[0]
            if self._load_tools_or_fail():
                self._send_json(_research_call("list_hf_models", repo_id=repo_id))

        elif path == "/hf/model-card":
            repo_id = qs.get("repo_id", [""])[0]
            revision = qs.get("revision", ["main"])[0]
            if self._load_tools_or_fail():
                self._send_json(
                    _research_call(
                        "fetch_hf_model_card", repo_id=repo_id, revision=revision,
                    ),
                )

        elif path == "/hf/artifact":
            filename = qs.get("filename", [""])[0]
            if not filename:
                self._send_json(
                    {"error": "Missing required query parameter: 'filename'"}, status=400,
                )
                return
            repo_id = qs.get("repo_id", [""])[0]
            revision = qs.get("revision", ["main"])[0]
            if self._load_tools_or_fail():
                self._send_json(
                    _research_call(
                        "fetch_hf_artifact",
                        filename=filename, repo_id=repo_id, revision=revision,
                    ),
                )

        # ---- history / tested models
        elif path == "/history/runs":
            if self._load_tools_or_fail():
                self._send_json(_research_call("list_hippius_runs"))

        elif path.startswith("/history/run/"):
            run_id = path.split("/")[-1]
            if not run_id or not _SAFE_PATH_RE.match(run_id.replace("-", "")):
                self._send_json({"error": f"Invalid run_id: {run_id!r}"}, status=400)
                return
            if self._load_tools_or_fail():
                self._send_json(_research_call("load_hippius_run", run_id=run_id))

        elif path == "/history/experiments":
            raw_limit = qs.get("limit", ["50"])[0]
            limit, limit_err = _validate_positive_int(raw_limit, "limit")
            if limit_err:
                self._send_json({"error": limit_err}, status=400)
                return
            if self._load_tools_or_fail():
                self._send_json(_research_call("load_hippius_history", limit=limit))

        elif path == "/history/trackio":
            raw_limit = qs.get("limit", ["20"])[0]
            limit, limit_err = _validate_positive_int(raw_limit, "limit")
            if limit_err:
                self._send_json({"error": limit_err}, status=400)
                return
            order = qs.get("order", ["best"])[0]
            if order not in ("best", "recent", "worst"):
                self._send_json(
                    {"error": f"Invalid order: {order!r}. Use 'best', 'recent', or 'worst'"},
                    status=400,
                )
                return
            if self._load_tools_or_fail():
                self._send_json(
                    _research_call("fetch_experiment_runs", limit=limit, order=order),
                )

        # ---- agent design
        elif path == "/agents/list":
            if self._load_tools_or_fail():
                self._send_json(_research_call("list_agents"))

        elif path == "/agents/read":
            filename = qs.get("filename", [""])[0]
            if not filename:
                self._send_json(
                    {"error": "Missing required query parameter: 'filename'"}, status=400,
                )
                return
            if self._load_tools_or_fail():
                result = _research_call("read_agent", filename=filename)
                self._send_json(result)

        elif path == "/agents/prompts/list":
            if self._load_tools_or_fail():
                self._send_json(_research_call("list_agent_prompts"))

        elif path == "/agents/prompts/read":
            filename = qs.get("filename", [""])[0]
            if not filename:
                self._send_json(
                    {"error": "Missing required query parameter: 'filename'"}, status=400,
                )
                return
            if self._load_tools_or_fail():
                result = _research_call("read_agent_prompt", filename=filename)
                self._send_json(result)

        elif path == "/agents/tools":
            if self._load_tools_or_fail():
                self._send_json(_research_call("list_available_tools"))

        else:
            self._send_json({"error": f"Not found: {path}"}, status=404)

    # ---- POST routes
    def do_POST(self) -> None:
        if not self._check_auth():
            return
        path, _qs = self._path_parts()
        body = self._read_body()
        if body is None:
            return  # _read_body already sent the error response

        if path == "/pipeline/run":
            with _pipeline_lock:
                if _pipeline_state.running:
                    self._send_json(
                        {
                            "error": "Pipeline is already running",
                            "status": _pipeline_state.status,
                        },
                        status=409,
                    )
                    return
                _pipeline_state.reset()

            retries = body.get("retries", 5)
            temperature = body.get("temperature", 0.1)
            retries_v, retries_err = _validate_positive_int(retries, "retries")
            if retries_err:
                with _pipeline_lock:
                    _pipeline_state.mark_failed(ValueError(retries_err))
                self._send_json({"error": retries_err}, status=400)
                return
            temp_v, temp_err = _validate_positive_float(temperature, "temperature")
            if temp_err:
                with _pipeline_lock:
                    _pipeline_state.mark_failed(ValueError(temp_err))
                self._send_json({"error": temp_err}, status=400)
                return

            task = {
                "channel": body.get("channel", "default"),
                "retries": retries_v,
                "temperature": temp_v,
                "publish": bool(body.get("publish", False)),
            }
            thread = threading.Thread(
                target=_run_pipeline_async, args=(task,), daemon=True,
            )
            thread.start()
            self._send_json(
                {"status": "started", "message": "Pipeline running in background"},
            )

        elif path == "/experiment/create":
            # Validate before loading tools
            blocks = body.get("blocks", [])
            if not isinstance(blocks, list) or not blocks:
                self._send_json(
                    {"error": "'blocks' must be a non-empty list of block names"},
                    status=400,
                )
                return
            blocks_json = json.dumps(blocks)

            d_model = body.get("d_model", 32)
            d_val, d_err = _validate_positive_int(d_model, "d_model")
            if d_err:
                self._send_json({"error": d_err}, status=400)
                return

            if self._load_tools_or_fail():
                self._send_json(_research_call(
                    "create_experiment",
                    blocks=blocks_json,
                    head=body.get("head", "GBMHead"),
                    d_model=d_val,
                    horizon=body.get("horizon", 12),
                    n_paths=body.get("n_paths", 100),
                    lr=body.get("lr", 0.001),
                ))

        elif path == "/experiment/validate":
            experiment = body.get("experiment")
            if experiment is None:
                self._send_json(
                    {"error": "Missing required field: 'experiment'"}, status=400,
                )
                return
            if isinstance(experiment, dict):
                experiment = json.dumps(experiment)
            if self._load_tools_or_fail():
                self._send_json(
                    _research_call("validate_experiment", experiment=experiment),
                )

        elif path == "/experiment/run":
            experiment = body.get("experiment")
            if experiment is None:
                self._send_json(
                    {"error": "Missing required field: 'experiment'"}, status=400,
                )
                return
            if isinstance(experiment, dict):
                experiment = json.dumps(experiment)
            epochs = body.get("epochs", 1)
            epochs_v, epochs_err = _validate_positive_int(epochs, "epochs")
            if epochs_err:
                self._send_json({"error": epochs_err}, status=400)
                return
            name = body.get("name", "")
            if self._load_tools_or_fail():
                self._send_json(_research_call(
                    "run_experiment", experiment=experiment, epochs=epochs_v, name=name,
                ))

        elif path == "/session/clear":
            if self._load_tools_or_fail():
                self._send_json(_research_call("clear_session"))

        # ---- registry write / reload
        elif path == "/registry/write":
            filename = body.get("filename", "")
            code = body.get("code", "")
            if not filename:
                self._send_json(
                    {"error": "Missing required field: 'filename'"}, status=400,
                )
                return
            if not code:
                self._send_json(
                    {"error": "Missing required field: 'code'"}, status=400,
                )
                return
            if self._load_tools_or_fail():
                self._send_json(
                    _research_call("write_component", filename=filename, code=code),
                )

        elif path == "/registry/reload":
            if self._load_tools_or_fail():
                self._send_json(_research_call("reload_registry"))

        # ---- agent design (write)
        elif path == "/agents/write":
            filename = body.get("filename", "")
            code = body.get("code", "")
            if not filename:
                self._send_json(
                    {"error": "Missing required field: 'filename'"}, status=400,
                )
                return
            if not code:
                self._send_json(
                    {"error": "Missing required field: 'code'"}, status=400,
                )
                return
            if self._load_tools_or_fail():
                self._send_json(
                    _research_call("write_agent", filename=filename, code=code),
                )

        elif path == "/agents/prompts/write":
            filename = body.get("filename", "")
            code = body.get("code", "")
            if not filename:
                self._send_json(
                    {"error": "Missing required field: 'filename'"}, status=400,
                )
                return
            if not code:
                self._send_json(
                    {"error": "Missing required field: 'code'"}, status=400,
                )
                return
            if self._load_tools_or_fail():
                self._send_json(
                    _research_call("write_agent_prompt", filename=filename, code=code),
                )

        else:
            self._send_json({"error": f"Not found: {path}"}, status=404)


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def run_bridge(host: str = "127.0.0.1", port: int = 8377) -> None:
    """Start the bridge HTTP server."""
    from cli.display import console, print_banner

    server = HTTPServer((host, port), BridgeHandler)
    print_banner(subtitle="bridge server")
    logger.info("synth-city bridge listening on http://%s:%d", host, port)
    console.print(
        f"  [bold cyan]bridge listening on[/bold cyan] "
        f"[link=http://{host}:{port}]http://{host}:{port}[/link]"
    )
    console.print("  [muted]Press Ctrl+C to stop.[/muted]\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[warning]Shutting down bridge server.[/warning]")
    finally:
        server.server_close()
