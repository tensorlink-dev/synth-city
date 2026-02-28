"""Tests for the OpenClaw integration layer (bridge, client, skill tools)."""

from __future__ import annotations

import importlib
import json
import subprocess
import threading
import time
from http.server import HTTPServer
from typing import Any
from unittest import mock

import httpx
import pytest

import integrations.openclaw.bridge as bridge_mod
from integrations.openclaw.bot_sessions import PipelineState
from integrations.openclaw.bridge import (
    MAX_CONTENT_LENGTH,
    VALID_ASSETS,
    BridgeHandler,
    _validate_asset,
    _validate_positive_float,
    _validate_positive_int,
)
from integrations.openclaw.client import SynthCityClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _start_bridge(port: int = 0) -> tuple[HTTPServer, int, threading.Thread]:
    """Start a bridge server on an ephemeral port. Returns (server, port, thread)."""
    server = HTTPServer(("127.0.0.1", port), BridgeHandler)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, actual_port, thread


# ---------------------------------------------------------------------------
# PipelineState unit tests
# ---------------------------------------------------------------------------

class TestPipelineState:
    def test_initial_state(self) -> None:
        state = PipelineState()
        d = state.to_dict()
        assert d["running"] is False
        assert d["status"] == "idle"
        assert d["result"] is None
        assert d["error"] is None
        assert d["elapsed_seconds"] is None

    def test_reset(self) -> None:
        state = PipelineState()
        state.reset()
        assert state.running is True
        assert state.status == "running"
        assert state.started_at is not None
        assert state.current_stage == "initializing"

    def test_mark_completed(self) -> None:
        state = PipelineState()
        state.reset()
        state.mark_completed({"crps": 0.1})
        assert state.running is False
        assert state.status == "completed"
        assert state.result == {"crps": 0.1}
        assert state.finished_at is not None

    def test_mark_failed(self) -> None:
        state = PipelineState()
        state.reset()
        state.mark_failed(ValueError("bad input"))
        assert state.running is False
        assert state.status == "failed"
        assert state.error is not None
        assert "ValueError" in state.error
        assert "bad input" in state.error

    def test_elapsed_seconds_while_running(self) -> None:
        state = PipelineState()
        state.reset()
        time.sleep(0.05)
        d = state.to_dict()
        assert d["elapsed_seconds"] is not None
        assert d["elapsed_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# Validation helper tests
# ---------------------------------------------------------------------------

class TestValidationHelpers:
    def test_validate_asset_valid(self) -> None:
        for asset in VALID_ASSETS:
            assert _validate_asset(asset) is None

    def test_validate_asset_case_insensitive(self) -> None:
        assert _validate_asset("btc") is None
        assert _validate_asset("Eth") is None

    def test_validate_asset_invalid(self) -> None:
        assert _validate_asset("FAKE") is not None
        assert _validate_asset("") is not None
        assert _validate_asset("../etc") is not None

    def test_validate_positive_int(self) -> None:
        val, err = _validate_positive_int(5, "x")
        assert val == 5
        assert err is None

    def test_validate_positive_int_string(self) -> None:
        val, err = _validate_positive_int("10", "x")
        assert val == 10
        assert err is None

    def test_validate_positive_int_negative(self) -> None:
        val, err = _validate_positive_int(-1, "x")
        assert val is None
        assert err is not None and "positive" in err

    def test_validate_positive_int_zero(self) -> None:
        val, err = _validate_positive_int(0, "x")
        assert val is None
        assert err is not None and "positive" in err

    def test_validate_positive_int_bad_type(self) -> None:
        val, err = _validate_positive_int("abc", "x")
        assert val is None
        assert err is not None and "integer" in err

    def test_validate_positive_float(self) -> None:
        val, err = _validate_positive_float(0.5, "x")
        assert val == 0.5
        assert err is None

    def test_validate_positive_float_negative(self) -> None:
        val, err = _validate_positive_float(-0.1, "x")
        assert val is None
        assert err is not None and "positive" in err


# ---------------------------------------------------------------------------
# Bridge HTTP integration tests
# ---------------------------------------------------------------------------

class TestBridgeHTTP:
    """Test the bridge HTTP server using a real server on an ephemeral port."""

    @pytest.fixture(autouse=True)
    def _server(self) -> Any:
        """Start a bridge server for each test."""
        self.server, port, self.thread = _start_bridge()
        self.base_url = f"http://127.0.0.1:{port}"
        self.client = SynthCityClient(
            base_url=self.base_url,
            timeout=5.0,
            retries=1,
        )
        yield
        self.server.shutdown()

    def test_health(self) -> None:
        resp = self.client.health()
        assert resp["status"] == "ok"
        assert resp["service"] == "synth-city-bridge"

    def test_pipeline_status_idle(self) -> None:
        resp = self.client.pipeline_status()
        assert resp["status"] == "idle"
        assert resp["running"] is False

    def test_get_not_found(self) -> None:
        resp = httpx.get(f"{self.base_url}/nonexistent", timeout=5.0)
        data = resp.json()
        assert resp.status_code == 404
        assert "Not found" in data["error"]

    def test_post_not_found(self) -> None:
        resp = httpx.post(
            f"{self.base_url}/nonexistent",
            json={},
            timeout=5.0,
        )
        data = resp.json()
        assert resp.status_code == 404
        assert "Not found" in data["error"]

    def test_post_invalid_json(self) -> None:
        port = self.server.server_address[1]
        resp = httpx.post(
            f"http://127.0.0.1:{port}/experiment/create",
            content=b"not json",
            headers={"Content-Type": "application/json", "Content-Length": "8"},
            timeout=5.0,
        )
        data = resp.json()
        assert "error" in data
        assert "Invalid JSON" in data["error"]

    def test_max_content_length_configured(self) -> None:
        # httpx/h11 enforces Content-Length consistency client-side, so we
        # can't send a mismatched header in tests. Verify the constant is set.
        assert MAX_CONTENT_LENGTH == 1 * 1024 * 1024

    def test_post_non_object_body(self) -> None:
        port = self.server.server_address[1]
        body = b"[1, 2, 3]"
        resp = httpx.post(
            f"http://127.0.0.1:{port}/experiment/create",
            content=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            timeout=5.0,
        )
        data = resp.json()
        assert resp.status_code == 400
        assert "Expected JSON object" in data["error"]

    def test_experiment_create_missing_blocks(self) -> None:
        resp = httpx.post(
            f"{self.base_url}/experiment/create",
            json={},
            timeout=5.0,
        )
        data = resp.json()
        assert resp.status_code == 400
        assert "blocks" in data["error"].lower()

    def test_experiment_create_empty_blocks(self) -> None:
        resp = httpx.post(
            f"{self.base_url}/experiment/create",
            json={"blocks": []},
            timeout=5.0,
        )
        data = resp.json()
        assert resp.status_code == 400
        assert "blocks" in data["error"].lower()

    def test_experiment_validate_missing_experiment(self) -> None:
        resp = httpx.post(
            f"{self.base_url}/experiment/validate",
            json={},
            timeout=5.0,
        )
        data = resp.json()
        assert resp.status_code == 400
        assert "experiment" in data["error"].lower()

    def test_experiment_run_missing_experiment(self) -> None:
        resp = httpx.post(
            f"{self.base_url}/experiment/run",
            json={},
            timeout=5.0,
        )
        data = resp.json()
        assert resp.status_code == 400
        assert "experiment" in data["error"].lower()

    def test_experiment_run_invalid_epochs(self) -> None:
        resp = httpx.post(
            f"{self.base_url}/experiment/run",
            json={
                "experiment": {"blocks": ["TransformerBlock"]},
                "epochs": -1,
            },
            timeout=5.0,
        )
        data = resp.json()
        assert resp.status_code == 400
        assert "positive" in data["error"]

    def test_market_price_invalid_asset(self) -> None:
        resp = httpx.get(f"{self.base_url}/market/price/FAKE", timeout=5.0)
        data = resp.json()
        assert resp.status_code == 400
        assert "Unknown asset" in data["error"]

    def test_market_price_path_traversal(self) -> None:
        resp = httpx.get(f"{self.base_url}/market/price/../etc/passwd", timeout=5.0)
        data = resp.json()
        assert "error" in data

    def test_market_history_invalid_days(self) -> None:
        resp = httpx.get(
            f"{self.base_url}/market/history/BTC",
            params={"days": "abc"},
            timeout=5.0,
        )
        data = resp.json()
        assert resp.status_code == 400
        assert "integer" in data["error"]

    def test_pipeline_run_invalid_retries(self) -> None:
        resp = httpx.post(
            f"{self.base_url}/pipeline/run",
            json={"retries": "bad"},
            timeout=5.0,
        )
        data = resp.json()
        assert resp.status_code == 400
        assert "error" in data

    def test_pipeline_run_invalid_temperature(self) -> None:
        resp = httpx.post(
            f"{self.base_url}/pipeline/run",
            json={"temperature": -0.5},
            timeout=5.0,
        )
        data = resp.json()
        assert resp.status_code == 400
        assert "error" in data


# ---------------------------------------------------------------------------
# Client unit tests
# ---------------------------------------------------------------------------

class TestClient:
    def test_default_config_from_env(self) -> None:
        with mock.patch.dict("os.environ", {"SYNTH_BRIDGE_URL": "http://custom:9999"}):
            import integrations.openclaw.client as client_mod

            importlib.reload(client_mod)
            assert client_mod._DEFAULT_BRIDGE_URL == "http://custom:9999"
        # Restore outside of env patch
        importlib.reload(client_mod)

    def test_retry_on_connect_error(self) -> None:
        # Import fresh to avoid class identity issues from module reloads
        from integrations.openclaw.client import BridgeConnectionError

        client = SynthCityClient(
            base_url="http://127.0.0.1:1",  # port 1 — nothing listens here
            timeout=0.5,
            retries=2,
        )
        with pytest.raises(BridgeConnectionError) as exc_info:
            client.health()
        assert "unreachable" in str(exc_info.value).lower()
        assert "2 attempts" in str(exc_info.value)

    def test_strips_trailing_slash(self) -> None:
        client = SynthCityClient(base_url="http://localhost:8377/")
        assert client.base_url == "http://localhost:8377"


# ---------------------------------------------------------------------------
# Skill tools tests
# ---------------------------------------------------------------------------

class TestSkillTools:
    def test_bridge_url_from_env(self) -> None:
        with mock.patch.dict("os.environ", {"SYNTH_BRIDGE_URL": "http://remote:5000"}):
            import integrations.openclaw.skill.tools as tools_mod

            importlib.reload(tools_mod)
            assert tools_mod.BRIDGE_URL == "http://remote:5000"
        importlib.reload(tools_mod)

    def test_curl_not_found(self) -> None:
        from integrations.openclaw.skill.tools import _curl_get

        with mock.patch("integrations.openclaw.skill.tools.shutil.which", return_value=None):
            result = _curl_get("/health")
            data = json.loads(result)
            assert "error" in data
            assert "curl not found" in data["error"]

    def test_curl_timeout(self) -> None:
        from integrations.openclaw.skill.tools import _curl_get

        with mock.patch(
            "integrations.openclaw.skill.tools.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="curl", timeout=120),
        ), mock.patch(
            "integrations.openclaw.skill.tools.shutil.which", return_value="/usr/bin/curl",
        ):
            result = _curl_get("/health")
            data = json.loads(result)
            assert "error" in data
            assert "timed out" in data["error"].lower()

    def test_curl_post_timeout(self) -> None:
        from integrations.openclaw.skill.tools import _curl_post

        with mock.patch(
            "integrations.openclaw.skill.tools.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="curl", timeout=300),
        ), mock.patch(
            "integrations.openclaw.skill.tools.shutil.which", return_value="/usr/bin/curl",
        ):
            result = _curl_post("/pipeline/run", {})
            data = json.loads(result)
            assert "error" in data
            assert "timed out" in data["error"].lower()

    def test_curl_failure_includes_detail(self) -> None:
        from integrations.openclaw.skill.tools import _curl_get

        mock_result = mock.Mock()
        mock_result.returncode = 7
        mock_result.stderr = "Failed to connect"
        mock_result.stdout = ""

        with mock.patch(
            "integrations.openclaw.skill.tools.subprocess.run", return_value=mock_result,
        ), mock.patch(
            "integrations.openclaw.skill.tools.shutil.which", return_value="/usr/bin/curl",
        ):
            result = _curl_get("/health")
            data = json.loads(result)
            assert "error" in data
            assert "detail" in data
            assert "Failed to connect" in data["detail"]

    def test_auth_headers_empty_when_no_key(self) -> None:
        from integrations.openclaw.skill.tools import _auth_headers

        with mock.patch("integrations.openclaw.skill.tools.BRIDGE_API_KEY", ""):
            assert _auth_headers() == []

    def test_auth_headers_set_when_key_present(self) -> None:
        from integrations.openclaw.skill.tools import _auth_headers

        with mock.patch("integrations.openclaw.skill.tools.BRIDGE_API_KEY", "my-key"):
            assert _auth_headers() == ["-H", "X-API-Key: my-key"]

    def test_validate_experiment_bad_json(self) -> None:
        from integrations.openclaw.skill.tools import synth_validate_experiment

        result = synth_validate_experiment("not valid json")
        data = json.loads(result)
        assert "error" in data
        assert "Invalid experiment JSON" in data["error"]

    def test_run_experiment_bad_json(self) -> None:
        from integrations.openclaw.skill.tools import synth_run_experiment

        result = synth_run_experiment("{{bad")
        data = json.loads(result)
        assert "error" in data
        assert "Invalid experiment JSON" in data["error"]


# ---------------------------------------------------------------------------
# API key authentication tests
# ---------------------------------------------------------------------------

class TestBridgeAPIKeyAuth:
    """Test that the bridge enforces API key auth when BRIDGE_API_KEY is set."""

    @pytest.fixture(autouse=True)
    def _server_with_key(self) -> Any:
        """Start a bridge with API key enabled."""
        self._original_key = bridge_mod.BRIDGE_API_KEY
        bridge_mod.BRIDGE_API_KEY = "test-secret-key"
        self.server, port, self.thread = _start_bridge()
        self.base_url = f"http://127.0.0.1:{port}"
        yield
        self.server.shutdown()
        bridge_mod.BRIDGE_API_KEY = self._original_key

    def test_request_without_key_returns_401(self) -> None:
        resp = httpx.get(f"{self.base_url}/health", timeout=5.0)
        assert resp.status_code == 401
        data = resp.json()
        assert "API key" in data["error"]

    def test_request_with_wrong_key_returns_401(self) -> None:
        resp = httpx.get(
            f"{self.base_url}/health",
            headers={"X-API-Key": "wrong-key"},
            timeout=5.0,
        )
        assert resp.status_code == 401

    def test_request_with_correct_key_succeeds(self) -> None:
        resp = httpx.get(
            f"{self.base_url}/health",
            headers={"X-API-Key": "test-secret-key"},
            timeout=5.0,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_post_without_key_returns_401(self) -> None:
        resp = httpx.post(
            f"{self.base_url}/session/clear",
            json={},
            timeout=5.0,
        )
        assert resp.status_code == 401

    def test_post_with_correct_key_succeeds(self) -> None:
        resp = httpx.get(
            f"{self.base_url}/pipeline/status",
            headers={"X-API-Key": "test-secret-key"},
            timeout=5.0,
        )
        assert resp.status_code == 200

    def test_client_sends_api_key(self) -> None:
        client = SynthCityClient(
            base_url=self.base_url,
            timeout=5.0,
            retries=1,
            api_key="test-secret-key",
        )
        resp = client.health()
        assert resp["status"] == "ok"

    def test_client_without_key_fails(self) -> None:
        client = SynthCityClient(
            base_url=self.base_url,
            timeout=5.0,
            retries=1,
            api_key="",
        )
        resp = client.health()
        assert "error" in resp


class TestBridgeNoAPIKey:
    """Test that auth is disabled when BRIDGE_API_KEY is empty."""

    @pytest.fixture(autouse=True)
    def _server_no_key(self) -> Any:
        self._original_key = bridge_mod.BRIDGE_API_KEY
        bridge_mod.BRIDGE_API_KEY = ""
        self.server, port, self.thread = _start_bridge()
        self.base_url = f"http://127.0.0.1:{port}"
        yield
        self.server.shutdown()
        bridge_mod.BRIDGE_API_KEY = self._original_key

    def test_request_without_key_succeeds(self) -> None:
        resp = httpx.get(f"{self.base_url}/health", timeout=5.0)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# File locks pruning tests
# ---------------------------------------------------------------------------

class TestFileLocksPruning:
    """Test that _file_locks dict is pruned when it exceeds the cap."""

    def test_locks_pruned_when_cap_reached(self) -> None:
        from pathlib import Path

        import pipeline.tools.file_tools as ft

        original_locks = ft._file_locks
        original_max = ft._MAX_FILE_LOCKS
        try:
            ft._file_locks = {}
            ft._MAX_FILE_LOCKS = 5

            # Fill up to the cap
            for i in range(5):
                ft._get_file_lock(Path(f"/fake/path/{i}.txt"))
            assert len(ft._file_locks) == 5

            # Adding one more should trigger pruning of unheld locks
            ft._get_file_lock(Path("/fake/path/new.txt"))
            # All prior locks were unheld so they should be pruned;
            # only the new lock remains
            assert len(ft._file_locks) == 1
            assert "/fake/path/new.txt" in str(list(ft._file_locks.keys())[0])
        finally:
            ft._file_locks = original_locks
            ft._MAX_FILE_LOCKS = original_max

    def test_held_locks_survive_pruning(self) -> None:
        from pathlib import Path

        import pipeline.tools.file_tools as ft

        original_locks = ft._file_locks
        original_max = ft._MAX_FILE_LOCKS
        try:
            ft._file_locks = {}
            ft._MAX_FILE_LOCKS = 3

            # Create and hold one lock
            held = ft._get_file_lock(Path("/fake/held.txt"))
            held.acquire()

            ft._get_file_lock(Path("/fake/a.txt"))
            ft._get_file_lock(Path("/fake/b.txt"))
            assert len(ft._file_locks) == 3

            # Trigger pruning — held lock should survive
            ft._get_file_lock(Path("/fake/c.txt"))
            assert "/fake/held.txt" in str(list(ft._file_locks.keys()))
            held.release()
        finally:
            ft._file_locks = original_locks
            ft._MAX_FILE_LOCKS = original_max


# ---------------------------------------------------------------------------
# _ensure_tools_loaded thread safety tests
# ---------------------------------------------------------------------------

class TestEnsureToolsLoaded:
    """Test double-checked locking in _ensure_tools_loaded."""

    def test_lock_exists(self) -> None:
        """The module-level lock used for double-checked locking must exist."""
        assert hasattr(bridge_mod, "_tools_load_lock")
        assert isinstance(bridge_mod._tools_load_lock, type(threading.Lock()))

    def test_idempotent_when_already_loaded(self) -> None:
        """If _tools_loaded is True, the function returns immediately."""
        original = bridge_mod._tools_loaded
        try:
            bridge_mod._tools_loaded = True
            # Should return without attempting any imports
            bridge_mod._ensure_tools_loaded()
            assert bridge_mod._tools_loaded is True
        finally:
            bridge_mod._tools_loaded = original

    def test_concurrent_calls_serialize(self) -> None:
        """Multiple threads should serialize through the lock without error."""
        original_loaded = bridge_mod._tools_loaded
        try:
            # Pre-set as loaded so concurrent calls hit the fast path
            bridge_mod._tools_loaded = True

            barrier = threading.Barrier(4)
            errors: list[Exception] = []

            def load() -> None:
                barrier.wait()
                try:
                    bridge_mod._ensure_tools_loaded()
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=load) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert bridge_mod._tools_loaded is True
            assert not errors, f"Unexpected errors: {errors}"
        finally:
            bridge_mod._tools_loaded = original_loaded

    def test_failed_load_leaves_flag_false(self) -> None:
        """If imports fail, _tools_loaded stays False so retry is possible."""
        original_loaded = bridge_mod._tools_loaded
        try:
            bridge_mod._tools_loaded = False

            with mock.patch(
                "pipeline.tools.agent_tools",
                create=True,
                side_effect=ImportError("test"),
            ):
                # Patch the actual import to raise
                import builtins
                original_import = builtins.__import__

                def failing_import(name, *args, **kwargs):
                    if name == "pipeline.tools.agent_tools":
                        raise ImportError("test failure")
                    return original_import(name, *args, **kwargs)

                builtins.__import__ = failing_import
                try:
                    with pytest.raises(ImportError):
                        bridge_mod._ensure_tools_loaded()
                    assert bridge_mod._tools_loaded is False
                finally:
                    builtins.__import__ = original_import
        finally:
            bridge_mod._tools_loaded = original_loaded


# ---------------------------------------------------------------------------
# Bot session active-request tracking tests
# ---------------------------------------------------------------------------

class TestBotSessionActiveRequests:
    """Test in-flight request tracking on BotSession."""

    def test_acquire_release_cycle(self) -> None:
        from integrations.openclaw.bot_sessions import BotSession

        session = BotSession(bot_id="test-req")
        assert session.has_active_requests() is False

        session.acquire_request()
        assert session.has_active_requests() is True

        session.release_request()
        assert session.has_active_requests() is False

    def test_multiple_concurrent_requests(self) -> None:
        from integrations.openclaw.bot_sessions import BotSession

        session = BotSession(bot_id="test-multi")
        session.acquire_request()
        session.acquire_request()
        assert session.has_active_requests() is True

        session.release_request()
        assert session.has_active_requests() is True  # still one in-flight

        session.release_request()
        assert session.has_active_requests() is False

    def test_release_does_not_go_negative(self) -> None:
        from integrations.openclaw.bot_sessions import BotSession

        session = BotSession(bot_id="test-neg")
        session.release_request()  # no prior acquire
        assert session.has_active_requests() is False

    def test_set_and_reset_session_tracks_requests(self) -> None:
        from integrations.openclaw.bot_sessions import (
            BotSession,
            reset_current_session,
            set_current_session,
        )

        session = BotSession(bot_id="test-ctx")
        assert session.has_active_requests() is False

        token = set_current_session(session)
        assert session.has_active_requests() is True

        reset_current_session(token)
        assert session.has_active_requests() is False

    def test_reaper_skips_session_with_active_requests(self) -> None:
        from integrations.openclaw.bot_sessions import SessionRegistry

        reg = SessionRegistry(ttl_seconds=0)  # immediate expiry
        session = reg.get_or_create("active-bot")
        session.acquire_request()
        session.last_active = 0  # force it to look expired

        # Simulate reaper check
        now = time.time()
        to_remove: list[str] = []
        with reg._lock:
            for bot_id, s in reg._sessions.items():
                if bot_id == "default":
                    continue
                if s.pipeline_state.running:
                    continue
                if s.has_active_requests():
                    continue
                if now - s.last_active > reg._ttl:
                    to_remove.append(bot_id)

        assert "active-bot" not in to_remove
        session.release_request()


# ---------------------------------------------------------------------------
# Setup tests
# ---------------------------------------------------------------------------

class TestSetup:
    def test_install_creates_skill_dir(self, tmp_path: Any) -> None:
        from integrations.openclaw.setup import install_skill

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        result = install_skill(workspace)
        if result:
            skill_dir = workspace / "skills" / "synth-city"
            assert skill_dir.exists()
            assert (skill_dir / "SKILL.md").exists()
            assert (skill_dir / "tools.py").exists()

    def test_install_idempotent(self, tmp_path: Any) -> None:
        from integrations.openclaw.setup import install_skill

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Install twice — should not fail
        result1 = install_skill(workspace)
        result2 = install_skill(workspace)
        assert result1 == result2
