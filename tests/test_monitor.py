"""Tests for pipeline.monitor — event collector singleton."""

from __future__ import annotations

import threading
import time

from pipeline.monitor import Monitor


def _fresh_monitor() -> Monitor:
    """Return a brand-new Monitor (not the global singleton)."""
    return Monitor()


class TestEmitAndSnapshot:
    def test_initial_snapshot_is_idle(self) -> None:
        mon = _fresh_monitor()
        snap = mon.snapshot()
        assert snap["status"] == "idle"
        assert snap["experiments_run"] == 0
        assert snap["best_crps"] is None
        assert snap["crps_history"] == []
        assert snap["errors"] == []

    def test_pipeline_start_sets_running(self) -> None:
        mon = _fresh_monitor()
        mon.emit("pipeline", "pipeline_start", stages=4)
        snap = mon.snapshot()
        assert snap["status"] == "running"
        assert snap["total_stages"] == 4
        assert snap["started_at"] is not None

    def test_stage_start_updates_stage(self) -> None:
        mon = _fresh_monitor()
        mon.emit("pipeline", "pipeline_start", stages=4)
        mon.emit("pipeline", "stage_start", stage="trainer", stage_num=2)
        snap = mon.snapshot()
        assert snap["current_stage"] == "trainer"
        assert snap["current_stage_num"] == 2

    def test_retry_attempt_updates_temperature(self) -> None:
        mon = _fresh_monitor()
        mon.emit("pipeline", "retry_attempt", attempt=2, max_attempts=5, temperature=0.3)
        snap = mon.snapshot()
        assert snap["current_attempt"] == 2
        assert snap["temperature"] == 0.3

    def test_agent_lifecycle(self) -> None:
        mon = _fresh_monitor()
        mon.emit(
            "agent", "agent_start",
            name="trainer", model="Qwen3", max_turns=50, tools=["a", "b"],
        )
        snap = mon.snapshot()
        assert snap["agent"]["name"] == "trainer"
        assert snap["agent"]["model"] == "Qwen3"
        assert snap["agent"]["max_turns"] == 50
        assert snap["agent"]["tools"] == ["a", "b"]

        mon.emit("agent", "agent_turn", turn=5)
        assert mon.snapshot()["agent"]["turn"] == 5

        mon.emit("agent", "agent_finish", success=True, turns=10)
        assert mon.snapshot()["agent"]["turn"] == 10

    def test_experiment_result_tracks_crps(self) -> None:
        mon = _fresh_monitor()
        mon.emit("experiment", "experiment_result", name="exp1", crps=0.05)
        mon.emit("experiment", "experiment_result", name="exp2", crps=0.03)
        mon.emit("experiment", "experiment_result", name="exp3", crps=0.07)

        snap = mon.snapshot()
        assert snap["experiments_run"] == 3
        assert snap["best_crps"] == 0.03
        assert snap["best_experiment_name"] == "exp2"
        assert len(snap["crps_history"]) == 3

    def test_tool_call_and_result(self) -> None:
        mon = _fresh_monitor()
        mon.emit("tool", "tool_call", name="run_experiment")
        snap = mon.snapshot()
        assert len(snap["recent_tool_calls"]) == 1
        assert snap["recent_tool_calls"][0]["status"] == "running"

        mon.emit("tool", "tool_result", name="run_experiment", size=1234)
        snap = mon.snapshot()
        assert snap["recent_tool_calls"][0]["status"] == "done"
        assert snap["recent_tool_calls"][0]["size"] == 1234

    def test_stall_detected(self) -> None:
        mon = _fresh_monitor()
        mon.emit("pipeline", "stall_detected")
        assert mon.snapshot()["stall_detected"] is True

    def test_pipeline_complete_success(self) -> None:
        mon = _fresh_monitor()
        mon.emit("pipeline", "pipeline_start", stages=3)
        mon.emit("pipeline", "pipeline_complete", success=True, best_crps=0.04)
        snap = mon.snapshot()
        assert snap["status"] == "completed"

    def test_pipeline_complete_failure(self) -> None:
        mon = _fresh_monitor()
        mon.emit("pipeline", "pipeline_start", stages=3)
        mon.emit("pipeline", "pipeline_complete", success=False)
        assert mon.snapshot()["status"] == "failed"

    def test_error_collection(self) -> None:
        mon = _fresh_monitor()
        mon.emit("system", "error", message="something broke")
        assert "something broke" in mon.snapshot()["errors"]


class TestRingBuffer:
    def test_overflow_caps_at_max(self) -> None:
        mon = _fresh_monitor()
        for i in range(600):
            mon.emit("test", "event", index=i)
        events = mon.events_since(0)
        assert len(events) == 500
        # Oldest events should be gone; newest should be present
        assert events[0]["data"]["index"] == 100
        assert events[-1]["data"]["index"] == 599


class TestEventsSince:
    def test_filters_by_timestamp(self) -> None:
        mon = _fresh_monitor()
        mon.emit("a", "e1")
        cutoff = time.time()
        time.sleep(0.01)
        mon.emit("b", "e2")
        mon.emit("c", "e3")

        after = mon.events_since(cutoff)
        assert len(after) == 2
        assert after[0]["event_type"] == "e2"
        assert after[1]["event_type"] == "e3"

    def test_returns_empty_when_none_match(self) -> None:
        mon = _fresh_monitor()
        mon.emit("a", "e1")
        future = time.time() + 1000
        assert mon.events_since(future) == []


class TestReset:
    def test_reset_clears_everything(self) -> None:
        mon = _fresh_monitor()
        mon.emit("pipeline", "pipeline_start", stages=4)
        mon.emit("experiment", "experiment_result", name="x", crps=0.05)
        mon.emit("system", "error", message="oops")

        mon.reset()
        snap = mon.snapshot()
        assert snap["status"] == "idle"
        assert snap["experiments_run"] == 0
        assert snap["best_crps"] is None
        assert snap["errors"] == []
        assert mon.events_since(0) == []


class TestThreadSafety:
    def test_concurrent_emits(self) -> None:
        mon = _fresh_monitor()
        barrier = threading.Barrier(4)

        def worker(cat: str, count: int) -> None:
            barrier.wait()
            for i in range(count):
                mon.emit(cat, f"event_{i}", index=i)

        threads = [
            threading.Thread(target=worker, args=("a", 50)),
            threading.Thread(target=worker, args=("b", 50)),
            threading.Thread(target=worker, args=("c", 50)),
            threading.Thread(target=worker, args=("d", 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        events = mon.events_since(0)
        assert len(events) == 200  # 4 * 50


class TestPipelineCompleteClearsAgentState:
    def test_pipeline_complete_clears_agent_and_stage(self) -> None:
        mon = _fresh_monitor()
        mon.emit("pipeline", "pipeline_start", stages=3)
        mon.emit(
            "agent", "agent_start",
            name="trainer", model="Qwen3", max_turns=50, tools=["a"],
        )
        mon.emit("agent", "agent_turn", turn=3)
        mon.emit("pipeline", "pipeline_complete", success=True)
        snap = mon.snapshot()
        assert snap["status"] == "completed"
        assert snap["current_stage"] == ""
        assert snap["agent"]["name"] == ""
        assert snap["agent"]["turn"] == 0


class TestExperimentStartNoop:
    def test_experiment_start_does_not_alter_snapshot(self) -> None:
        mon = _fresh_monitor()
        snap_before = mon.snapshot()
        mon.emit("experiment", "experiment_start", name="exp1")
        snap_after = mon.snapshot()
        assert snap_after["experiments_run"] == snap_before["experiments_run"]
        assert snap_after["best_crps"] == snap_before["best_crps"]


class TestToolResultNoMatch:
    def test_tool_result_with_no_matching_call(self) -> None:
        mon = _fresh_monitor()
        # tool_result with no prior tool_call should not crash
        mon.emit("tool", "tool_result", name="nonexistent", size=99)
        snap = mon.snapshot()
        assert snap["recent_tool_calls"] == []

    def test_tool_result_with_already_done_call(self) -> None:
        mon = _fresh_monitor()
        mon.emit("tool", "tool_call", name="run_experiment")
        mon.emit("tool", "tool_result", name="run_experiment", size=100)
        # Second result for same tool — should not update the already-done entry
        mon.emit("tool", "tool_result", name="run_experiment", size=200)
        snap = mon.snapshot()
        assert snap["recent_tool_calls"][0]["size"] == 100


class TestDequeOverflow:
    def test_crps_history_caps_at_max(self) -> None:
        mon = _fresh_monitor()
        for i in range(150):
            mon.emit(
                "experiment", "experiment_result",
                name=f"exp{i}", crps=float(i),
            )
        snap = mon.snapshot()
        assert len(snap["crps_history"]) == 100
        # Oldest should be gone; newest should be present
        assert snap["crps_history"][0]["name"] == "exp50"
        assert snap["crps_history"][-1]["name"] == "exp149"
        # experiments_run still counts all
        assert snap["experiments_run"] == 150

    def test_errors_caps_at_max(self) -> None:
        mon = _fresh_monitor()
        for i in range(70):
            mon.emit("system", "error", message=f"err{i}")
        snap = mon.snapshot()
        assert len(snap["errors"]) == 50
        assert snap["errors"][0] == "err20"
        assert snap["errors"][-1] == "err69"


class TestExperimentResultWithoutCrps:
    def test_no_crps_still_increments_count(self) -> None:
        mon = _fresh_monitor()
        mon.emit("experiment", "experiment_result", name="exp_nocrps")
        snap = mon.snapshot()
        assert snap["experiments_run"] == 1
        assert snap["best_crps"] is None
        assert snap["crps_history"] == []


class TestElapsed:
    def test_elapsed_seconds_when_running(self) -> None:
        mon = _fresh_monitor()
        mon.emit("pipeline", "pipeline_start", stages=3)
        time.sleep(0.05)
        snap = mon.snapshot()
        assert snap["elapsed_seconds"] >= 0.04

    def test_elapsed_zero_when_idle(self) -> None:
        mon = _fresh_monitor()
        assert mon.snapshot()["elapsed_seconds"] == 0.0

    def test_elapsed_zero_when_completed(self) -> None:
        mon = _fresh_monitor()
        mon.emit("pipeline", "pipeline_start", stages=3)
        time.sleep(0.02)
        mon.emit("pipeline", "pipeline_complete", success=True)
        snap = mon.snapshot()
        assert snap["status"] == "completed"
        assert snap["elapsed_seconds"] == 0.0
