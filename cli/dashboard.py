"""Rich Live terminal dashboard for real-time pipeline monitoring.

Provides a full-screen auto-refreshing display that shows:
  - Pipeline stage progress and retry state
  - Current agent status (model, turn, tools)
  - Experiment results ranked by CRPS
  - Live activity feed
  - Alerts (stalls, errors)

Usage::

    # Local mode: runs the pipeline with live dashboard overlay
    synth-city dashboard

    # Remote mode: polls a running bridge server
    synth-city dashboard --remote http://127.0.0.1:8377
"""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.display import console

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_STAGE_LABELS = {
    "planner": "PLANNER",
    "trainer": "TRAINER",
    "check_debug": "CHECK / DEBUG",
    "codechecker": "CODECHECKER",
    "debugger": "DEBUGGER",
    "publisher": "PUBLISHER",
}

_STATUS_STYLE = {
    "idle": ("dim", "IDLE"),
    "running": ("bold green", "RUNNING"),
    "completed": ("bold cyan", "COMPLETED"),
    "failed": ("bold red", "FAILED"),
}

_CAT_STYLE = {
    "pipeline": "bold cyan",
    "agent": "bold magenta",
    "tool": "yellow",
    "experiment": "bold green",
    "system": "bold red",
}


def _fmt_elapsed(seconds: float) -> str:
    if seconds <= 0:
        return "--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _fmt_ts(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def _pipeline_panel(snap: dict[str, Any]) -> Panel:
    """Build the pipeline status panel."""
    style_key, status_label = _STATUS_STYLE.get(snap["status"], ("dim", snap["status"]))
    stage = _STAGE_LABELS.get(snap["current_stage"], snap["current_stage"] or "--")
    stage_num = snap.get("current_stage_num", 0)
    total = snap.get("total_stages", 0)

    lines = Text()
    lines.append("  Status:   ", style="bold white")
    lines.append(f"{status_label}\n", style=style_key)
    lines.append("  Stage:    ", style="bold white")
    lines.append(f"{stage_num}/{total} {stage}\n", style="cyan")
    lines.append("  Attempt:  ", style="bold white")
    lines.append(f"{snap.get('current_attempt', 0)}/{snap.get('max_attempts', 0)}\n")
    lines.append("  Temp:     ", style="bold white")
    lines.append(f"{snap.get('temperature', 0):.2f}\n")
    lines.append("  Elapsed:  ", style="bold white")
    lines.append(f"{_fmt_elapsed(snap.get('elapsed_seconds', 0))}\n")
    lines.append("  Best CRPS:", style="bold white")
    crps = snap.get("best_crps")
    if crps is not None:
        lines.append(f" {crps:.6f}", style="bold magenta")
    else:
        lines.append(" --", style="dim")

    return Panel(lines, title="[bold cyan]Pipeline[/bold cyan]", border_style="cyan")


def _agent_panel(snap: dict[str, Any]) -> Panel:
    """Build the current agent status panel."""
    agent = snap.get("agent", {})
    name = agent.get("name", "--")
    # Shorten long model names to last segment
    short_name = name.rsplit("/", 1)[-1] if "/" in name else name

    lines = Text()
    lines.append("  Name:   ", style="bold white")
    lines.append(f"{short_name}\n", style="magenta")
    lines.append("  Model:  ", style="bold white")
    lines.append(f"{agent.get('model', '--')}\n", style="dim")
    lines.append("  Turn:   ", style="bold white")
    lines.append(f"{agent.get('turn', 0)}/{agent.get('max_turns', 0)}\n")
    lines.append("  Tools:  ", style="bold white")
    lines.append(f"{len(agent.get('tools', []))}")

    return Panel(lines, title="[bold magenta]Agent[/bold magenta]", border_style="magenta")


def _experiments_panel(snap: dict[str, Any]) -> Panel:
    """Build the experiments ranking table."""
    table = Table(
        show_header=True, header_style="bold cyan",
        border_style="cyan", pad_edge=True, expand=True,
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Name", style="white", min_width=15, no_wrap=True)
    table.add_column("CRPS", style="bold magenta", justify="right", width=10)

    history = snap.get("crps_history", [])
    # Sort by CRPS ascending (best first)
    ranked = sorted(history, key=lambda x: x.get("crps", float("inf")))
    for i, entry in enumerate(ranked[:10], 1):
        crps = entry.get("crps")
        crps_str = f"{crps:.6f}" if isinstance(crps, (int, float)) else "N/A"
        table.add_row(str(i), entry.get("name", "?"), crps_str)

    if not ranked:
        table.add_row("", "[dim]No experiments yet[/dim]", "")

    count = snap.get("experiments_run", 0)
    return Panel(
        table,
        title=f"[bold cyan]Experiments ({count} run)[/bold cyan]",
        border_style="cyan",
    )


def _activity_panel(events: list[dict[str, Any]]) -> Panel:
    """Build the recent activity feed."""
    lines = Text()
    # Show most recent first, limit to 15
    for ev in reversed(events[-15:]):
        ts = _fmt_ts(ev.get("timestamp", 0))
        cat = ev.get("category", "?")
        etype = ev.get("event_type", "?")
        data = ev.get("data", {})
        style = _CAT_STYLE.get(cat, "dim")

        lines.append(f"  {ts} ", style="dim")
        lines.append(f"{cat:10s} ", style=style)

        # Format the event detail
        if etype == "stage_start":
            label = _STAGE_LABELS.get(data.get("stage", ""), data.get("stage", ""))
            lines.append(f"STAGE {data.get('stage_num', '?')}: {label}")
        elif etype == "agent_turn":
            lines.append(f"turn {data.get('turn', '?')}")
        elif etype == "tool_call":
            lines.append(f"{data.get('name', '?')} called")
        elif etype == "tool_result":
            size = data.get("size", 0)
            lines.append(f"{data.get('name', '?')} -> {size:,} chars")
        elif etype == "experiment_result":
            crps = data.get("crps")
            crps_s = f"{crps:.6f}" if isinstance(crps, (int, float)) else "N/A"
            lines.append(f"{data.get('name', '?')} CRPS={crps_s}")
        elif etype == "retry_attempt":
            lines.append(
                f"attempt {data.get('attempt')}/{data.get('max_attempts')} "
                f"temp={data.get('temperature', 0):.2f}"
            )
        elif etype == "stall_detected":
            lines.append("STALL DETECTED", style="bold red")
        elif etype == "pipeline_start":
            lines.append("pipeline started")
        elif etype == "pipeline_complete":
            ok = data.get("success", False)
            lines.append(
                "pipeline completed" if ok else "pipeline FAILED",
                style="green" if ok else "red",
            )
        else:
            lines.append(etype)

        lines.append("\n")

    if not events:
        lines.append("  Waiting for events...", style="dim")

    return Panel(lines, title="[bold yellow]Activity[/bold yellow]", border_style="yellow")


def _alerts_panel(snap: dict[str, Any]) -> Panel:
    """Build the alerts/warnings panel."""
    lines = Text()
    if snap.get("stall_detected"):
        lines.append("  STALL DETECTED — experiment config unchanged\n", style="bold red")
    for err in snap.get("errors", [])[-5:]:
        lines.append(f"  {err}\n", style="red")
    if not snap.get("stall_detected") and not snap.get("errors"):
        lines.append("  (none)", style="dim")
    return Panel(lines, title="[bold red]Alerts[/bold red]", border_style="red")


# ---------------------------------------------------------------------------
# Layout assembly
# ---------------------------------------------------------------------------

def build_dashboard_layout(snap: dict[str, Any], events: list[dict[str, Any]]) -> Layout:
    """Assemble the full dashboard layout from a snapshot and event list."""
    layout = Layout()
    layout.split_column(
        Layout(name="top", size=10),
        Layout(name="middle", size=14),
        Layout(name="bottom"),
    )
    layout["top"].split_row(
        Layout(_pipeline_panel(snap), name="pipeline"),
        Layout(_agent_panel(snap), name="agent"),
    )
    layout["middle"].update(_experiments_panel(snap))
    layout["bottom"].split_row(
        Layout(_activity_panel(events), name="activity", ratio=3),
        Layout(_alerts_panel(snap), name="alerts", ratio=1),
    )
    return layout


# ---------------------------------------------------------------------------
# Dashboard runners
# ---------------------------------------------------------------------------

def run_dashboard(
    task: dict[str, Any],
    max_retries: int = 5,
    base_temperature: float = 0.1,
    publish: bool = False,
) -> dict[str, Any]:
    """Run the pipeline in a background thread with a Rich Live dashboard overlay.

    Returns the pipeline result dict.
    """
    from pipeline.bootstrap import bootstrap_all
    from pipeline.monitor import get_monitor
    from pipeline.orchestrator import PipelineOrchestrator

    bootstrap_all()
    mon = get_monitor()
    result_holder: dict[str, Any] = {}

    def _pipeline_thread() -> None:
        orchestrator = PipelineOrchestrator(
            max_retries=max_retries,
            base_temperature=base_temperature,
            publish=publish,
        )
        result_holder["result"] = orchestrator.run(task)

    thread = threading.Thread(target=_pipeline_thread, daemon=True)
    thread.start()

    with Live(console=console, refresh_per_second=1, screen=False) as live:
        last_ts = 0.0
        all_events: list[dict[str, Any]] = []
        while thread.is_alive():
            snap = mon.snapshot()
            new_events = mon.events_since(last_ts)
            if new_events:
                all_events.extend(new_events)
                last_ts = new_events[-1]["timestamp"]
                # Cap to prevent unbounded memory growth
                if len(all_events) > 500:
                    all_events = all_events[-500:]
            live.update(build_dashboard_layout(snap, all_events))
            time.sleep(1.0)

        # Final render after pipeline finishes
        snap = mon.snapshot()
        all_events.extend(mon.events_since(last_ts))
        live.update(build_dashboard_layout(snap, all_events))

    return result_holder.get("result", {})


def run_dashboard_remote(bridge_url: str) -> None:
    """Poll a remote bridge server and display the dashboard.

    Press Ctrl+C to stop.
    """
    snapshot_url = f"{bridge_url.rstrip('/')}/dash/snapshot"
    events_url = f"{bridge_url.rstrip('/')}/dash/events"

    with Live(console=console, refresh_per_second=1, screen=False) as live:
        last_ts = 0.0
        all_events: list[dict[str, Any]] = []
        try:
            while True:
                try:
                    snap = httpx.get(snapshot_url, timeout=5).json()
                    new = httpx.get(
                        events_url, params={"after": str(last_ts)}, timeout=5,
                    ).json()
                    if isinstance(new, list) and new:
                        all_events.extend(new)
                        last_ts = new[-1].get("timestamp", last_ts)
                        # Cap to prevent unbounded memory growth
                        if len(all_events) > 500:
                            all_events = all_events[-500:]
                except Exception:
                    snap = {"status": "disconnected", "errors": ["Bridge unreachable"]}
                    new = []

                live.update(build_dashboard_layout(snap, all_events))
                time.sleep(2.0)
        except KeyboardInterrupt:
            console.print("\n[warning]Dashboard stopped.[/warning]")
