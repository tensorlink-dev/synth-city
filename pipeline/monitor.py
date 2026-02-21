"""
Dashboard monitor — thread-safe event collector for real-time pipeline observability.

Provides a singleton ``Monitor`` that captures structured events from the
orchestrator, agent loop, and tool execution.  Both the Rich Live terminal
dashboard and the web GUI poll this object for current state.

Usage::

    from pipeline.monitor import get_monitor

    mon = get_monitor()
    mon.emit("pipeline", "stage_start", stage="planner", stage_num=1)

    snapshot = mon.snapshot()   # dict — current pipeline state
    events   = mon.events_since(ts)  # list[dict] — events after *ts*
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DashboardEvent:
    """A single observable event emitted by the pipeline."""

    timestamp: float
    category: str       # pipeline | agent | tool | experiment | system
    event_type: str     # stage_start, agent_turn, tool_call, experiment_result, …
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "category": self.category,
            "event_type": self.event_type,
            "data": self.data,
        }


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

_MAX_EVENTS = 500
_MAX_TOOL_CALLS = 20
_MAX_CRPS_HISTORY = 100


class Monitor:
    """Thread-safe pipeline event collector with ring buffer.

    Maintains a rolling window of events and a live snapshot of the
    pipeline's current state.  All public methods are safe to call from
    any thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: deque[DashboardEvent] = deque(maxlen=_MAX_EVENTS)
        self._reset_snapshot()

    # ---- snapshot state (protected by _lock) ----

    def _reset_snapshot(self) -> None:
        """Initialise / reset all mutable snapshot fields."""
        self._status: str = "idle"
        self._current_stage: str = ""
        self._current_stage_num: int = 0
        self._total_stages: int = 0
        self._current_attempt: int = 0
        self._max_attempts: int = 0
        self._temperature: float = 0.0

        self._agent_name: str = ""
        self._agent_model: str = ""
        self._agent_turn: int = 0
        self._agent_max_turns: int = 0
        self._agent_tools: list[str] = []

        self._experiments_run: int = 0
        self._best_crps: float | None = None
        self._best_experiment_name: str = ""
        self._crps_history: list[dict[str, Any]] = []

        self._recent_tool_calls: deque[dict[str, Any]] = deque(maxlen=_MAX_TOOL_CALLS)

        self._started_at: float | None = None
        self._stall_detected: bool = False
        self._errors: list[str] = []

    # ---- public API ----

    def emit(self, category: str, event_type: str, **data: Any) -> None:
        """Record an event and update the live snapshot.

        This is fire-and-forget — it never raises and never blocks the
        pipeline for longer than the lock acquisition.
        """
        now = time.time()
        event = DashboardEvent(
            timestamp=now,
            category=category,
            event_type=event_type,
            data=dict(data),
        )

        with self._lock:
            self._events.append(event)
            self._update_snapshot(event)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the current pipeline state."""
        with self._lock:
            elapsed = 0.0
            if self._started_at and self._status == "running":
                elapsed = round(time.time() - self._started_at, 1)

            return {
                "status": self._status,
                "current_stage": self._current_stage,
                "current_stage_num": self._current_stage_num,
                "total_stages": self._total_stages,
                "current_attempt": self._current_attempt,
                "max_attempts": self._max_attempts,
                "temperature": self._temperature,
                "agent": {
                    "name": self._agent_name,
                    "model": self._agent_model,
                    "turn": self._agent_turn,
                    "max_turns": self._agent_max_turns,
                    "tools": list(self._agent_tools),
                },
                "experiments_run": self._experiments_run,
                "best_crps": self._best_crps,
                "best_experiment_name": self._best_experiment_name,
                "crps_history": list(self._crps_history[-_MAX_CRPS_HISTORY:]),
                "recent_tool_calls": list(self._recent_tool_calls),
                "started_at": self._started_at,
                "elapsed_seconds": elapsed,
                "stall_detected": self._stall_detected,
                "errors": list(self._errors[-20:]),
            }

    def events_since(self, after: float) -> list[dict[str, Any]]:
        """Return all events with ``timestamp > after``."""
        with self._lock:
            return [
                e.to_dict() for e in self._events
                if e.timestamp > after
            ]

    def reset(self) -> None:
        """Clear all events and reset the snapshot for a new pipeline run."""
        with self._lock:
            self._events.clear()
            self._reset_snapshot()

    # ---- internal snapshot updater ----

    def _update_snapshot(self, event: DashboardEvent) -> None:  # noqa: C901
        """Mutate snapshot fields based on the incoming event.

        Called under ``_lock`` — must be fast and never raise.
        """
        etype = event.event_type
        data = event.data

        # -- pipeline events --
        if event.category == "pipeline":
            if etype == "pipeline_start":
                self._status = "running"
                self._started_at = event.timestamp
                self._total_stages = data.get("stages", 4)
                self._stall_detected = False
                self._errors.clear()

            elif etype == "stage_start":
                self._current_stage = data.get("stage", "")
                self._current_stage_num = data.get("stage_num", 0)
                self._current_attempt = 0

            elif etype == "retry_attempt":
                self._current_attempt = data.get("attempt", 0)
                self._max_attempts = data.get("max_attempts", 0)
                self._temperature = data.get("temperature", 0.0)

            elif etype == "stall_detected":
                self._stall_detected = True

            elif etype == "pipeline_complete":
                self._status = "completed" if data.get("success") else "failed"
                self._current_stage = ""
                self._agent_name = ""
                self._agent_turn = 0

        # -- agent events --
        elif event.category == "agent":
            if etype == "agent_start":
                self._agent_name = data.get("name", "")
                self._agent_model = data.get("model", "")
                self._agent_max_turns = data.get("max_turns", 0)
                self._agent_tools = data.get("tools", [])
                self._agent_turn = 0

            elif etype == "agent_turn":
                self._agent_turn = data.get("turn", 0)

            elif etype == "agent_finish":
                self._agent_turn = data.get("turns", self._agent_turn)

        # -- tool events --
        elif event.category == "tool":
            if etype == "tool_call":
                self._recent_tool_calls.append({
                    "timestamp": event.timestamp,
                    "name": data.get("name", ""),
                    "status": "running",
                })
            elif etype == "tool_result":
                # Update the last matching call
                name = data.get("name", "")
                for tc in reversed(self._recent_tool_calls):
                    if tc.get("name") == name and tc.get("status") == "running":
                        tc["status"] = "done"
                        tc["size"] = data.get("size", 0)
                        break

        # -- experiment events --
        elif event.category == "experiment":
            if etype == "experiment_start":
                pass  # placeholder for future use

            elif etype == "experiment_result":
                self._experiments_run += 1
                crps = data.get("crps")
                name = data.get("name", "")
                if crps is not None:
                    self._crps_history.append({
                        "name": name,
                        "crps": crps,
                        "timestamp": event.timestamp,
                    })
                    if self._best_crps is None or crps < self._best_crps:
                        self._best_crps = crps
                        self._best_experiment_name = name

        # -- system events --
        elif event.category == "system":
            if etype == "error":
                msg = data.get("message", str(data))
                self._errors.append(msg)
            elif etype == "auto_flush":
                pass  # informational


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_monitor: Monitor | None = None
_monitor_lock = threading.Lock()


def get_monitor() -> Monitor:
    """Return the global Monitor singleton (created on first call)."""
    global _monitor
    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                _monitor = Monitor()
    return _monitor
