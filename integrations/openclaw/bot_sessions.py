"""
Bot session management for multi-clawbot concurrent access.

Each bot (identified by ``X-Bot-Id`` header) gets an isolated ``BotSession``
containing its own ResearchSession, run ID, PipelineState, and workspace
directory.  A ``SessionRegistry`` manages all active sessions and a background
reaper thread cleans up idle ones.

Tools access the current bot session via ``get_current_session()`` which reads
a ``contextvars.ContextVar`` set per-request by the bridge server.  When no
session is active (e.g. CLI mode), tools fall back to their module-level
defaults for backward compatibility.

Usage in the bridge::

    from integrations.openclaw.bot_sessions import registry, set_current_session

    session = registry.get_or_create(bot_id)
    token = set_current_session(session)
    try:
        handle_request(...)
    finally:
        reset_current_session(token)
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import BOT_SESSION_TTL_SECONDS, MAX_CONCURRENT_PIPELINES, WORKSPACE_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context variable — set per-request, read by tools
# ---------------------------------------------------------------------------

_current_bot: contextvars.ContextVar[BotSession | None] = contextvars.ContextVar(
    "current_bot_session", default=None,
)


def get_current_session() -> BotSession | None:
    """Return the active bot session for this context, or ``None`` in CLI mode."""
    return _current_bot.get()


def set_current_session(session: BotSession) -> contextvars.Token:
    """Set the bot session for the current context.  Returns a reset token."""
    session.acquire_request()
    return _current_bot.set(session)


def reset_current_session(token: contextvars.Token) -> None:
    """Restore the previous context after a request completes."""
    session = _current_bot.get()
    if session is not None:
        session.release_request()
    _current_bot.reset(token)


# ---------------------------------------------------------------------------
# Pipeline state (per-bot)
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """Tracks the status of an asynchronous pipeline run for one bot."""

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


# ---------------------------------------------------------------------------
# Bot session
# ---------------------------------------------------------------------------

@dataclass
class BotSession:
    """Isolated state for a single bot."""

    bot_id: str
    pipeline_state: PipelineState = field(default_factory=PipelineState)
    pipeline_lock: threading.Lock = field(default_factory=threading.Lock)
    run_id: str = ""
    workspace_dir: Path = field(default_factory=lambda: WORKSPACE_DIR)
    last_active: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

    # Lazily initialised per-bot ResearchSession
    _research_session: Any = field(default=None, repr=False)
    _research_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Track in-flight HTTP requests so the reaper doesn't evict active sessions
    _active_requests: int = field(default=0, repr=False)
    _active_requests_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not self.run_id:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            short_id = self.bot_id[:12] if self.bot_id != "default" else ""
            id_part = f"-{short_id}" if short_id else ""
            self.run_id = f"{ts}{id_part}-{uuid.uuid4().hex[:8]}"
        if self.bot_id != "default":
            self.workspace_dir = WORKSPACE_DIR / self.bot_id
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def touch(self) -> None:
        """Update last-active timestamp."""
        self.last_active = time.time()

    def acquire_request(self) -> None:
        """Mark the start of an in-flight HTTP request."""
        with self._active_requests_lock:
            self._active_requests += 1
        self.touch()

    def release_request(self) -> None:
        """Mark the end of an in-flight HTTP request."""
        self.touch()
        with self._active_requests_lock:
            self._active_requests = max(0, self._active_requests - 1)

    def has_active_requests(self) -> bool:
        """Return ``True`` if any HTTP requests are currently in-flight."""
        with self._active_requests_lock:
            return self._active_requests > 0

    def get_research_session(self):
        """Return (or lazily create) the per-bot ResearchSession."""
        if self._research_session is not None:
            return self._research_session
        with self._research_lock:
            # Double-check after acquiring lock
            if self._research_session is None:
                from src.research.agent_api import ResearchSession
                self._research_session = ResearchSession()
            return self._research_session

    def reset_run_id(self) -> None:
        """Generate a fresh run ID for the next pipeline invocation."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        short_id = self.bot_id[:12] if self.bot_id != "default" else ""
        id_part = f"-{short_id}" if short_id else ""
        self.run_id = f"{ts}{id_part}-{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bot_id": self.bot_id,
            "run_id": self.run_id,
            "workspace_dir": str(self.workspace_dir),
            "created_at": self.created_at,
            "last_active": self.last_active,
            "idle_seconds": round(time.time() - self.last_active, 1),
            "pipeline": self.pipeline_state.to_dict(),
        }


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------

class SessionRegistry:
    """Thread-safe registry of bot sessions with TTL-based cleanup."""

    def __init__(self, ttl_seconds: int = BOT_SESSION_TTL_SECONDS) -> None:
        self._sessions: dict[str, BotSession] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._pipeline_semaphore = threading.Semaphore(MAX_CONCURRENT_PIPELINES)
        self._reaper_started = False

    @property
    def pipeline_semaphore(self) -> threading.Semaphore:
        """Global semaphore to cap total concurrent pipeline runs."""
        return self._pipeline_semaphore

    def get_or_create(self, bot_id: str) -> BotSession:
        """Get an existing session or create a new one."""
        with self._lock:
            session = self._sessions.get(bot_id)
            if session is None:
                session = BotSession(bot_id=bot_id)
                self._sessions[bot_id] = session
                logger.info("Created bot session: %s", bot_id)
            session.touch()
            self._maybe_start_reaper()
            return session

    def get(self, bot_id: str) -> BotSession | None:
        """Get a session without creating one."""
        with self._lock:
            return self._sessions.get(bot_id)

    def remove(self, bot_id: str) -> bool:
        """Remove a session.  Returns True if it existed."""
        with self._lock:
            session = self._sessions.pop(bot_id, None)
            if session:
                logger.info("Removed bot session: %s", bot_id)
                return True
            return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return a snapshot of all active sessions."""
        with self._lock:
            return [s.to_dict() for s in self._sessions.values()]

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _maybe_start_reaper(self) -> None:
        """Start the background reaper thread (once)."""
        if self._reaper_started:
            return
        self._reaper_started = True
        t = threading.Thread(target=self._reaper_loop, daemon=True, name="bot-reaper")
        t.start()

    def _reaper_loop(self) -> None:
        """Periodically evict idle sessions."""
        while True:
            time.sleep(min(self._ttl / 2, 300))
            now = time.time()
            to_remove: list[str] = []
            with self._lock:
                for bot_id, session in self._sessions.items():
                    if bot_id == "default":
                        continue  # never evict the default session
                    if session.pipeline_state.running:
                        continue  # don't evict while pipeline is active
                    if session.has_active_requests():
                        continue  # don't evict while HTTP requests are in-flight
                    if now - session.last_active > self._ttl:
                        to_remove.append(bot_id)
                for bot_id in to_remove:
                    del self._sessions[bot_id]
                    logger.info("Reaped idle bot session: %s (TTL=%ds)", bot_id, self._ttl)


# Module-level singleton registry
registry = SessionRegistry()
