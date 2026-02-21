# Dashboard & TUI Monitoring — Implementation Plan

## Overview

Add two monitoring interfaces for synth-city's pipeline processes:

1. **Terminal Dashboard** — Rich Live display in the terminal (`synth-city dashboard`)
2. **Web GUI Dashboard** — Self-contained HTML page served by the bridge (`/dashboard`)

Both consume the same data: an **event collector** that hooks into the orchestrator, agent loop, and tool execution without heavy refactoring. Updates via polling (bridge `/dashboard/events` endpoint).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Event Collector                     │
│  (pipeline/monitor.py — thread-safe ring buffer)     │
│                                                       │
│  Hooks into:                                          │
│    • PipelineOrchestrator — stage transitions         │
│    • SimpleAgent — turn starts, tool calls, finishes  │
│    • ResearchSession — experiment results (CRPS)      │
└──────────┬──────────────────────────┬────────────────┘
           │                          │
    ┌──────▼──────┐          ┌───────▼────────┐
    │  Terminal    │          │  Bridge API     │
    │  Dashboard   │          │  /dashboard/*   │
    │  (Rich Live) │          │  endpoints      │
    │              │          │                 │
    │  cli/        │          │  → HTML page    │
    │  dashboard.py│          │  → JSON events  │
    └─────────────┘          └────────────────┘
```

---

## New Files

### 1. `pipeline/monitor.py` — Event Collector (core)
Thread-safe singleton that collects structured events from all processes.

```python
# Data model
@dataclass
class DashboardEvent:
    timestamp: float
    category: str        # "pipeline", "agent", "tool", "experiment", "system"
    event_type: str      # "stage_start", "turn", "tool_call", "crps_result", etc.
    data: dict[str, Any] # Payload (agent name, CRPS score, tool name, etc.)

@dataclass
class PipelineSnapshot:
    """Current state of the pipeline at a point in time."""
    run_id: str
    status: str              # "idle", "running", "completed", "failed"
    current_stage: str       # "planner", "trainer", "codechecker", "debugger", "publisher"
    current_stage_num: int   # 1-4
    total_stages: int        # 3 or 4
    current_attempt: int     # Retry attempt number
    max_attempts: int
    temperature: float       # Current LLM temperature

    # Agent state
    agent_name: str
    agent_model: str
    agent_turn: int
    agent_max_turns: int
    agent_tools: list[str]

    # Experiment metrics
    experiments_run: int
    best_crps: float | None
    best_experiment_name: str
    crps_history: list[dict]  # [{name, crps, timestamp}, ...]

    # Tool activity
    recent_tool_calls: list[dict]  # Last 20 tool calls

    # Timing
    started_at: float | None
    elapsed_seconds: float

    # Alerts
    stall_detected: bool
    errors: list[str]

class Monitor:
    """Singleton event collector with ring buffer."""
    _events: deque[DashboardEvent]  # Ring buffer, max 500 events
    _snapshot: PipelineSnapshot     # Current state
    _lock: threading.Lock

    def emit(category, event_type, **data)  # Record an event
    def snapshot() -> PipelineSnapshot      # Get current state
    def events_since(after: float) -> list  # Get events after timestamp
    def reset()                             # Clear for new pipeline run
```

**Key design decisions:**
- Singleton accessed via `get_monitor()` — no need to thread instances through the codebase
- Ring buffer (500 events max) prevents memory growth
- Lock-protected for thread safety (bridge runs in separate thread)
- `emit()` is fire-and-forget — never blocks pipeline execution

### 2. `cli/dashboard.py` — Terminal Dashboard
Rich Live display with auto-refreshing panels.

**Layout:**
```
┌─ SYNTH CITY DASHBOARD ──────────────────────────────┐
│                                                       │
│  ┌─ Pipeline ─────────┐  ┌─ Current Agent ─────────┐ │
│  │ Stage: 2/4 TRAINER  │  │ Name: trainer            │ │
│  │ Attempt: 1/5        │  │ Model: Qwen3-235B        │ │
│  │ Temp: 0.10          │  │ Turn: 7/50               │ │
│  │ Elapsed: 2m 34s     │  │ Tools: 12                │ │
│  │ Status: ● RUNNING   │  │ Status: ● ACTIVE         │ │
│  └─────────────────────┘  └─────────────────────────┘ │
│                                                       │
│  ┌─ Experiments ────────────────────────────────────┐ │
│  │ # │ Name              │ CRPS     │ Blocks        │ │
│  │ 1 │ exp_wavenet_gbm   │ 0.04231  │ WaveNet       │ │
│  │ 2 │ exp_lstm_flow     │ 0.04558  │ LSTM          │ │
│  │ 3 │ exp_tcn_gbm       │ 0.04892  │ TCN           │ │
│  └──────────────────────────────────────────────────┘ │
│                                                       │
│  ┌─ Recent Activity ───────────────────────────────┐  │
│  │ 14:23:05 tool   run_experiment → 1,243 chars     │  │
│  │ 14:23:02 tool   create_experiment → 892 chars    │  │
│  │ 14:22:58 agent  trainer turn 7                   │  │
│  │ 14:22:51 tool   list_blocks → 2,104 chars        │  │
│  │ 14:22:45 stage  STAGE 2: TRAINER started         │  │
│  └──────────────────────────────────────────────────┘ │
│                                                       │
│  ┌─ Alerts ────────────────────────────────────────┐  │
│  │ (none)                                           │  │
│  └──────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────┘
```

**Implementation:**
- Uses `rich.live.Live` with 1-second refresh
- Reads from `Monitor.snapshot()` and `Monitor.events_since()`
- Runs in a background thread alongside the pipeline
- Activated via `synth-city dashboard` CLI command (runs pipeline + live display)
- Alternatively `synth-city dashboard --remote` polls the bridge for remote monitoring

### 3. `dashboard/index.html` — Web GUI Dashboard
Single self-contained HTML file (inline CSS + JS, no build step).

**Layout:**
- Header: SYNTH CITY logo, connection status indicator, run ID
- Left column: Pipeline stage progress (vertical stepper), agent status card
- Center: CRPS chart (sparkline or mini bar chart via canvas), experiment table
- Right column: Live event feed (scrolling), alerts panel
- Footer: elapsed time, experiment count, best CRPS

**Implementation:**
- Vanilla JS with `fetch()` polling every 2 seconds
- CSS Grid layout, dark theme matching Rich terminal colors
- `<canvas>` for CRPS trend mini-chart (no external charting lib)
- Auto-reconnect on connection loss
- Responsive — works on mobile for remote monitoring

### 4. Bridge Endpoints (added to `integrations/openclaw/bridge.py`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/dashboard` | Serve the HTML dashboard page |
| GET | `/dashboard/snapshot` | Current pipeline state as JSON |
| GET | `/dashboard/events?after=<timestamp>` | Events since timestamp |

---

## Modifications to Existing Files

### `pipeline/orchestrator.py` — Add monitor hooks
Insert `monitor.emit()` calls at stage transitions:
- `run()`: emit `pipeline_start`, `pipeline_complete`
- Before each stage: emit `stage_start` with stage name/number
- `_run_with_retry()`: emit `retry_attempt` with attempt/temperature
- `_check_debug_loop()`: emit `stall_detected` on stall
- After `_save_to_hippius()`: emit `pipeline_saved`

~15 lines added (one `emit()` per event). No structural changes.

### `pipeline/providers/simple_agent.py` — Add monitor hooks
Insert `monitor.emit()` calls in the agent loop:
- `run()` start: emit `agent_start` with model, tools, max_turns
- Per turn: emit `agent_turn` with turn number
- `_execute_tool_call()`: emit `tool_call` with name, and `tool_result` with size
- `run()` finish: emit `agent_finish` with success, turns_used

~10 lines added. No structural changes.

### `pipeline/tools/research_tools.py` — Add experiment result hooks
After each experiment runs, emit `experiment_result` with name, CRPS, config.
~3 lines added.

### `cli/app.py` — Add `dashboard` subcommand
Add `cmd_dashboard()` handler and parser entry. ~30 lines.

### `integrations/openclaw/bridge.py` — Add dashboard routes
Add 3 GET routes (`/dashboard`, `/dashboard/snapshot`, `/dashboard/events`).
Add `_serve_dashboard_html()` to read and serve the HTML file.
~40 lines added.

---

## Dependencies

**None required.** Everything uses existing dependencies:
- `rich` (already installed) — Rich Live, Layout, Table, Panel, Text
- `threading`, `collections.deque`, `time`, `dataclasses` — stdlib
- Bridge already uses `http.server` — just add routes

---

## Implementation Order

### Phase 1: Event Collector (`pipeline/monitor.py`)
- Build the `Monitor` singleton with ring buffer
- Build `DashboardEvent` and `PipelineSnapshot` dataclasses
- Unit test: emit events, check snapshot, test `events_since()`

### Phase 2: Instrument Existing Code
- Add `monitor.emit()` hooks to `orchestrator.py` (6-8 calls)
- Add `monitor.emit()` hooks to `simple_agent.py` (4-5 calls)
- Add `monitor.emit()` hooks to `research_tools.py` (1-2 calls)
- Verify: run pipeline, confirm events are collected

### Phase 3: Terminal Dashboard (`cli/dashboard.py`)
- Build Rich Live layout with panels
- Add `cmd_dashboard` to CLI
- Test: run `synth-city dashboard` and observe live updates

### Phase 4: Web Dashboard
- Create `dashboard/index.html` with polling JS
- Add bridge endpoints (`/dashboard`, `/dashboard/snapshot`, `/dashboard/events`)
- Test: start bridge, open browser, observe live updates

### Phase 5: Polish
- Add CRPS trend sparkline to both dashboards
- Add elapsed time formatting
- Error/alert display
- Test remote monitoring mode (terminal dashboard polling bridge)

---

## Verification

1. **Unit test**: `tests/test_monitor.py` — test event emission, ring buffer overflow, snapshot, events_since
2. **Integration**: Run `synth-city pipeline` — verify monitor collects events without affecting pipeline behavior
3. **Terminal**: Run `synth-city dashboard` — verify live display updates in real time
4. **Web**: Start bridge, visit `http://localhost:8377/dashboard` — verify page loads and polls correctly
5. **Lint**: `ruff check .` passes
6. **Types**: `mypy pipeline/monitor.py cli/dashboard.py` passes
