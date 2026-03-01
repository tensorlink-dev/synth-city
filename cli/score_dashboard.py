"""Rich-powered display for the scoring emulator.

Provides:
    - Live dashboard when running ``synth-city score run``
    - Formatted tables for ``status``, ``results``, ``daily``, ``leaderboard``
"""

from __future__ import annotations

import time
from typing import Any

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.display import console, section_header

# ---------------------------------------------------------------------------
# Static output helpers (for status/results/daily/leaderboard)
# ---------------------------------------------------------------------------


def print_score_status(status: dict[str, Any]) -> None:
    """Print scoring emulator status."""
    section_header("Scoring Emulator Status")

    lines = Text()
    lines.append("  Prompts recorded:  ", style="bold white")
    lines.append(f"{status.get('prompts_recorded', 0)}\n")
    lines.append("  Prompts scored:    ", style="bold white")
    lines.append(f"{status.get('prompts_scored', 0)}\n", style="bold green")
    lines.append("  Prompts failed:    ", style="bold white")
    lines.append(f"{status.get('prompts_failed', 0)}\n", style="bold red")
    lines.append("  Pending:           ", style="bold white")
    lines.append(f"{status.get('pending_prompts', 0)}\n", style="bold yellow")
    lines.append("  Best weighted CRPS:", style="bold white")
    best = status.get("best_weighted_crps", float("inf"))
    if best < float("inf"):
        lines.append(f" {best:.4f}\n", style="bold magenta")
    else:
        lines.append(" --\n", style="dim")

    panel = Panel(lines, title="[bold cyan]Score Tracker[/bold cyan]", border_style="cyan")
    console.print(panel)

    # Pending details
    pending = status.get("pending_details", [])
    if pending:
        table = Table(
            title="Pending Prompts",
            border_style="yellow",
            header_style="bold yellow",
        )
        table.add_column("Prompt ID", style="white", min_width=25)
        table.add_column("Status", style="cyan")
        table.add_column("Age (min)", style="dim", justify="right")
        table.add_column("Assets", style="dim")
        table.add_column("Scored", style="green")

        for p in pending:
            table.add_row(
                p["prompt_id"],
                p["status"],
                f"{p['age_minutes']:.1f}",
                ", ".join(p["assets"]),
                ", ".join(p["scored_assets"]),
            )
        console.print(table)


def print_prompt_results(results: list[dict[str, Any]], date: str | None = None) -> None:
    """Print scored prompt results."""
    label = date or "today"
    section_header(f"Prompt Scores ({label})")

    if not results:
        console.print("[dim]  No scored prompts found.[/dim]")
        return

    table = Table(
        border_style="cyan",
        header_style="bold cyan",
        pad_edge=True,
    )
    table.add_column("#", style="dim", justify="right", width=4)
    table.add_column("Prompt ID", style="white", min_width=25)
    table.add_column("Model", style="dim", max_width=15)
    table.add_column("Status", style="cyan", width=8)
    table.add_column("Assets", justify="right", width=7)
    table.add_column("Weighted CRPS", style="bold magenta", justify="right", width=14)

    for i, r in enumerate(results, 1):
        wcrps = r.get("weighted_crps")
        crps_str = f"{wcrps:.4f}" if isinstance(wcrps, (int, float)) else "--"
        status_style = {
            "scored": "bold green",
            "partial": "bold yellow",
            "pending": "dim",
            "error": "bold red",
        }.get(r.get("status", ""), "dim")

        table.add_row(
            str(i),
            r.get("prompt_id", "?"),
            r.get("model_name", "?"),
            Text(r.get("status", "?"), style=status_style),
            str(len(r.get("assets_predicted", []))),
            crps_str,
        )
    console.print(table)

    # Show per-asset breakdown for the best prompt
    scored = [r for r in results if r.get("weighted_crps") is not None]
    if scored:
        best = min(scored, key=lambda r: r["weighted_crps"])
        _print_asset_breakdown(best)


def _print_asset_breakdown(prompt: dict[str, Any]) -> None:
    """Print per-asset CRPS breakdown for a prompt."""
    scores = prompt.get("scores", {})
    if not scores:
        return

    table = Table(
        title=f"Best Prompt: {prompt.get('prompt_id', '?')}",
        border_style="magenta",
        header_style="bold magenta",
    )
    table.add_column("Asset", style="white", width=10)
    table.add_column("CRPS 5m", justify="right", width=10)
    table.add_column("CRPS 1h", justify="right", width=10)
    table.add_column("CRPS 6h", justify="right", width=10)
    table.add_column("CRPS 24h", justify="right", width=10)
    table.add_column("Sum", style="bold magenta", justify="right", width=10)

    for asset, asset_scores in sorted(scores.items()):
        if not isinstance(asset_scores, dict):
            continue
        _fmt = lambda k: (  # noqa: E731
            f"{asset_scores[k]:.4f}" if k in asset_scores
            and isinstance(asset_scores[k], (int, float)) else "--"
        )
        table.add_row(
            asset,
            _fmt("crps_5min"),
            _fmt("crps_60min"),
            _fmt("crps_360min"),
            _fmt("crps_1440min"),
            _fmt("crps_sum"),
        )
    console.print(table)


def print_daily_summary(summary: dict[str, Any]) -> None:
    """Print a daily scoring summary."""
    section_header(f"Daily Summary — {summary.get('date', '?')}")

    lines = Text()
    lines.append("  Total prompts:   ", style="bold white")
    lines.append(f"{summary.get('total_prompts', 0)}\n")
    lines.append("  Scored:          ", style="bold white")
    lines.append(f"{summary.get('scored_prompts', 0)}\n", style="bold green")
    lines.append("  Pending:         ", style="bold white")
    lines.append(f"{summary.get('pending_prompts', 0)}\n", style="bold yellow")
    lines.append("  Failed:          ", style="bold white")
    lines.append(f"{summary.get('failed_prompts', 0)}\n", style="bold red")

    wc = summary.get("weighted_crps", {})
    if wc and wc.get("mean") is not None:
        lines.append("\n")
        lines.append("  Mean CRPS:       ", style="bold white")
        lines.append(f"{wc['mean']:.4f}\n", style="bold magenta")
        lines.append("  Best CRPS:       ", style="bold white")
        lines.append(f"{wc.get('best', 'N/A'):.4f}\n", style="bold green")
        lines.append("  Worst CRPS:      ", style="bold white")
        lines.append(f"{wc.get('worst', 'N/A'):.4f}\n", style="bold red")

    panel = Panel(lines, title="[bold cyan]Daily Scoring[/bold cyan]", border_style="cyan")
    console.print(panel)

    # Per-asset table
    per_asset = summary.get("per_asset", {})
    if per_asset:
        table = Table(
            title="Per-Asset Summary",
            border_style="cyan",
            header_style="bold cyan",
        )
        table.add_column("Asset", style="white", width=10)
        table.add_column("Prompts", justify="right", width=9)
        table.add_column("Mean CRPS", style="bold magenta", justify="right", width=12)
        table.add_column("Best CRPS", style="green", justify="right", width=12)
        table.add_column("Worst CRPS", style="red", justify="right", width=12)

        for asset, stats in sorted(per_asset.items()):
            _fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "--"  # noqa: E731
            table.add_row(
                asset,
                str(stats.get("prompt_count", 0)),
                _fmt(stats.get("mean_crps_sum")),
                _fmt(stats.get("best_crps_sum")),
                _fmt(stats.get("worst_crps_sum")),
            )
        console.print(table)


def print_leaderboard(lb: dict[str, Any]) -> None:
    """Print the rolling leaderboard."""
    section_header(f"Rolling Leaderboard ({lb.get('window_days', '?')}-day window)")

    if lb.get("status") == "no_data":
        console.print("[dim]  No scoring data available yet.[/dim]")
        console.print("[dim]  Run 'synth-city score run' to start the scoring emulator.[/dim]")
        return

    lines = Text()
    lines.append("  Window:          ", style="bold white")
    lines.append(f"{lb.get('window_days', '?')} days\n")
    lines.append("  Days with data:  ", style="bold white")
    lines.append(f"{lb.get('days_with_data', 0)}\n")
    lines.append("  Total prompts:   ", style="bold white")
    lines.append(f"{lb.get('total_prompts', 0)}\n")
    lines.append("  Scored prompts:  ", style="bold white")
    lines.append(f"{lb.get('scored_prompts', 0)}\n")

    wc = lb.get("weighted_crps", {})
    if wc and wc.get("rolling_mean") is not None:
        lines.append("\n")
        lines.append("  Rolling mean:    ", style="bold white")
        lines.append(f"{wc['rolling_mean']:.4f}\n", style="bold magenta")
        lines.append("  Rolling best:    ", style="bold white")
        lines.append(f"{wc.get('rolling_best', 'N/A'):.4f}\n", style="bold green")
        lines.append("  Rolling worst:   ", style="bold white")
        lines.append(f"{wc.get('rolling_worst', 'N/A'):.4f}\n", style="bold red")

    panel = Panel(lines, title="[bold cyan]Leaderboard[/bold cyan]", border_style="cyan")
    console.print(panel)

    # Per-asset rolling table
    per_asset = lb.get("per_asset", {})
    if per_asset:
        table = Table(
            title="Per-Asset Rolling Performance",
            border_style="cyan",
            header_style="bold cyan",
        )
        table.add_column("Asset", style="white", width=10)
        table.add_column("Prompts", justify="right", width=9)
        table.add_column("Mean CRPS", style="bold magenta", justify="right", width=12)
        table.add_column("Best Daily", style="green", justify="right", width=12)
        table.add_column("Trend", width=10)

        for asset, stats in sorted(per_asset.items()):
            mean_crps = stats.get("mean_crps")
            mean_str = f"{mean_crps:.4f}" if isinstance(mean_crps, (int, float)) else "--"
            best = stats.get("best_daily_crps")
            best_str = f"{best:.4f}" if isinstance(best, (int, float)) else "--"
            trend = stats.get("trend", "flat")
            trend_style = "green" if trend == "improving" else "dim"
            table.add_row(
                asset,
                str(stats.get("prompt_count", 0)),
                mean_str,
                best_str,
                Text(trend, style=trend_style),
            )
        console.print(table)

    updated = lb.get("updated_at", "")
    if updated:
        console.print(f"\n[dim]  Last updated: {updated}[/dim]")


# ---------------------------------------------------------------------------
# Live dashboard (for ``synth-city score run``)
# ---------------------------------------------------------------------------


def _score_pipeline_panel(status: dict[str, Any], daemon_running: bool) -> Panel:
    """Build the scoring daemon status panel."""
    lines = Text()
    lines.append("  Daemon:    ", style="bold white")
    if daemon_running:
        lines.append("RUNNING\n", style="bold green")
    else:
        lines.append("STOPPED\n", style="bold red")
    lines.append("  Recorded:  ", style="bold white")
    lines.append(f"{status.get('prompts_recorded', 0)}\n")
    lines.append("  Scored:    ", style="bold white")
    lines.append(f"{status.get('prompts_scored', 0)}\n", style="bold green")
    lines.append("  Pending:   ", style="bold white")
    lines.append(f"{status.get('pending_prompts', 0)}\n", style="bold yellow")
    lines.append("  Failed:    ", style="bold white")
    lines.append(f"{status.get('prompts_failed', 0)}\n", style="bold red")
    lines.append("  Best CRPS: ", style="bold white")
    best = status.get("best_weighted_crps", float("inf"))
    if best < float("inf"):
        lines.append(f"{best:.4f}", style="bold magenta")
    else:
        lines.append("--", style="dim")

    return Panel(
        lines,
        title="[bold cyan]Scoring Emulator[/bold cyan]",
        border_style="cyan",
    )


def _pending_prompts_panel(status: dict[str, Any]) -> Panel:
    """Build a panel showing pending prompts."""
    table = Table(
        show_header=True, header_style="bold yellow",
        border_style="yellow", pad_edge=True, expand=True,
    )
    table.add_column("Prompt ID", style="white", min_width=20, no_wrap=True)
    table.add_column("Status", style="cyan", width=8)
    table.add_column("Age", style="dim", justify="right", width=8)
    table.add_column("Assets", style="dim")

    for p in status.get("pending_details", [])[:8]:
        age = p.get("age_minutes", 0)
        if age >= 60:
            age_str = f"{age / 60:.1f}h"
        else:
            age_str = f"{age:.0f}m"
        table.add_row(
            p["prompt_id"],
            p["status"],
            age_str,
            ", ".join(p.get("assets", [])),
        )

    if not status.get("pending_details"):
        table.add_row("", "[dim]No pending prompts[/dim]", "", "")

    return Panel(
        table,
        title=f"[bold yellow]Pending ({status.get('pending_prompts', 0)})[/bold yellow]",
        border_style="yellow",
    )


def _recent_scores_panel(recent: list[dict[str, Any]]) -> Panel:
    """Build a panel showing recent scored prompts."""
    table = Table(
        show_header=True, header_style="bold magenta",
        border_style="magenta", pad_edge=True, expand=True,
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Prompt ID", style="white", min_width=20, no_wrap=True)
    table.add_column("Model", style="dim", max_width=12)
    table.add_column("Weighted CRPS", style="bold magenta", justify="right", width=14)
    table.add_column("Assets", style="dim", justify="right", width=7)

    for i, r in enumerate(recent[:10], 1):
        wcrps = r.get("weighted_crps")
        crps_str = f"{wcrps:.4f}" if isinstance(wcrps, (int, float)) else "--"
        table.add_row(
            str(i),
            r.get("prompt_id", "?"),
            r.get("model_name", "?"),
            crps_str,
            str(len(r.get("assets_predicted", []))),
        )

    if not recent:
        table.add_row("", "[dim]No scored prompts yet[/dim]", "", "", "")

    return Panel(
        table,
        title="[bold magenta]Recent Scores[/bold magenta]",
        border_style="magenta",
    )


def _build_score_layout(
    status: dict[str, Any],
    recent: list[dict[str, Any]],
    daemon_running: bool,
) -> Layout:
    """Build the full score dashboard layout."""
    layout = Layout()
    layout.split_column(
        Layout(name="top", size=10),
        Layout(name="bottom"),
    )
    layout["top"].split_row(
        Layout(_score_pipeline_panel(status, daemon_running), name="status"),
        Layout(_pending_prompts_panel(status), name="pending"),
    )
    layout["bottom"].update(_recent_scores_panel(recent))
    return layout


def run_score_dashboard(interval_minutes: int = 5) -> None:
    """Run the scoring daemon with a live Rich dashboard."""
    from cli.display import print_banner
    from subnet.score_tracker import ScoreTracker, ScoringDaemon

    print_banner(subtitle="scoring emulator")

    tracker = ScoreTracker()
    daemon = ScoringDaemon(tracker, interval_minutes=interval_minutes)
    daemon.start()

    console.print(
        f"[bold green]Scoring daemon started[/bold green] "
        f"(interval={interval_minutes}min). Press [bold]Ctrl+C[/bold] to stop.\n"
    )

    with Live(console=console, refresh_per_second=0.5, screen=False) as live:
        try:
            while daemon.running:
                status = tracker.get_status()
                recent = tracker.get_recent_scores(limit=10)
                live.update(_build_score_layout(status, recent, daemon.running))
                time.sleep(2.0)
        except KeyboardInterrupt:
            console.print("\n[warning]Stopping scoring daemon...[/warning]")
            daemon.stop()

    # Final summary
    console.print()
    section_header("Final Status")
    print_score_status(tracker.get_status())
