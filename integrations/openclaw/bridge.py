"""
OpenClaw ↔ synth-city bridge server.

Exposes synth-city's pipeline operations and research tools as a lightweight
HTTP API that OpenClaw workspace skills can call via ``fetch()`` or ``curl``.

Start with::

    python main.py bridge            # default: 127.0.0.1:8377
    python main.py bridge --port 9000

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
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


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
                round(time.time() - self.started_at, 1) if self.started_at and self.running else None
            ),
            "result": self.result,
            "error": self.error,
        }


_pipeline_state = PipelineState()
_pipeline_lock = threading.Lock()


def _run_pipeline_async(task: dict[str, Any]) -> None:
    """Execute the pipeline in a background thread."""
    global _pipeline_state
    try:
        from pipeline.orchestrator import PipelineOrchestrator

        with _pipeline_lock:
            _pipeline_state.running = True
            _pipeline_state.status = "running"
            _pipeline_state.started_at = time.time()
            _pipeline_state.finished_at = None
            _pipeline_state.result = None
            _pipeline_state.error = None
            _pipeline_state.current_stage = "initializing"

        orchestrator = PipelineOrchestrator(
            max_retries=task.get("retries", 5),
            base_temperature=task.get("temperature", 0.1),
            publish=task.get("publish", False),
        )
        result = orchestrator.run(task)

        with _pipeline_lock:
            _pipeline_state.running = False
            _pipeline_state.status = "completed"
            _pipeline_state.finished_at = time.time()
            _pipeline_state.result = result
            _pipeline_state.current_stage = ""

    except Exception as exc:
        with _pipeline_lock:
            _pipeline_state.running = False
            _pipeline_state.status = "failed"
            _pipeline_state.finished_at = time.time()
            _pipeline_state.error = f"{type(exc).__name__}: {exc}"
            _pipeline_state.current_stage = ""


# ---------------------------------------------------------------------------
# Research tool wrappers (lazy-loaded to avoid import overhead)
# ---------------------------------------------------------------------------

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
        return {"error": f"{type(exc).__name__}: {exc}"}


def _ensure_tools_loaded() -> None:
    """Import tool modules so they register with the tool registry."""
    import pipeline.tools.research_tools  # noqa: F401
    import pipeline.tools.market_data  # noqa: F401


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class BridgeHandler(BaseHTTPRequestHandler):
    """Routes HTTP requests to synth-city operations."""

    def log_message(self, format: str, *args: Any) -> None:
        logger.info(format, *args)

    # ---- helpers
    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, indent=2, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _path_parts(self) -> tuple[str, dict[str, list[str]]]:
        parsed = urlparse(self.path)
        return parsed.path.rstrip("/"), parse_qs(parsed.query)

    # ---- GET routes
    def do_GET(self) -> None:
        path, qs = self._path_parts()
        _ensure_tools_loaded()

        if path == "/health":
            self._send_json({"status": "ok", "service": "synth-city-bridge"})

        elif path == "/pipeline/status":
            self._send_json(_pipeline_state.to_dict())

        elif path == "/components/blocks":
            self._send_json(_research_call("list_blocks"))

        elif path == "/components/heads":
            self._send_json(_research_call("list_heads"))

        elif path == "/components/presets":
            self._send_json(_research_call("list_presets"))

        elif path == "/experiment/compare":
            self._send_json(_research_call("compare_results"))

        elif path == "/session/summary":
            self._send_json(_research_call("session_summary"))

        elif path.startswith("/market/price/"):
            asset = path.split("/")[-1]
            self._send_json(_research_call("get_latest_price", asset=asset))

        elif path.startswith("/market/history/"):
            asset = path.split("/")[-1]
            days = int(qs.get("days", ["30"])[0])
            self._send_json(_research_call("get_historical_data", asset=asset, days=days))

        else:
            self._send_json({"error": f"Not found: {path}"}, status=404)

    # ---- POST routes
    def do_POST(self) -> None:
        path, _qs = self._path_parts()
        body = self._read_body()
        _ensure_tools_loaded()

        if path == "/pipeline/run":
            if _pipeline_state.running:
                self._send_json(
                    {"error": "Pipeline is already running", "status": _pipeline_state.status},
                    status=409,
                )
                return

            task = {
                "channel": body.get("channel", "default"),
                "retries": body.get("retries", 5),
                "temperature": body.get("temperature", 0.1),
                "publish": body.get("publish", False),
            }
            thread = threading.Thread(target=_run_pipeline_async, args=(task,), daemon=True)
            thread.start()
            self._send_json({"status": "started", "message": "Pipeline running in background"})

        elif path == "/experiment/create":
            blocks = body.get("blocks", [])
            if isinstance(blocks, list):
                blocks = json.dumps(blocks)
            self._send_json(_research_call(
                "create_experiment",
                blocks=blocks,
                head=body.get("head", "GBMHead"),
                d_model=body.get("d_model", 32),
                horizon=body.get("horizon", 12),
                n_paths=body.get("n_paths", 100),
                lr=body.get("lr", 0.001),
            ))

        elif path == "/experiment/validate":
            experiment = body.get("experiment", "{}")
            if isinstance(experiment, dict):
                experiment = json.dumps(experiment)
            self._send_json(_research_call("validate_experiment", experiment=experiment))

        elif path == "/experiment/run":
            experiment = body.get("experiment", "{}")
            if isinstance(experiment, dict):
                experiment = json.dumps(experiment)
            epochs = body.get("epochs", 1)
            name = body.get("name", "")
            self._send_json(_research_call(
                "run_experiment", experiment=experiment, epochs=epochs, name=name,
            ))

        elif path == "/session/clear":
            self._send_json(_research_call("clear_session"))

        else:
            self._send_json({"error": f"Not found: {path}"}, status=404)


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def run_bridge(host: str = "127.0.0.1", port: int = 8377) -> None:
    """Start the bridge HTTP server."""
    server = HTTPServer((host, port), BridgeHandler)
    logger.info("synth-city bridge listening on http://%s:%d", host, port)
    print(f"synth-city bridge listening on http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down bridge server.")
        server.shutdown()
