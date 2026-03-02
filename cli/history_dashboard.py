"""Terminal dashboard for displaying historical experiment results.

Provides a Rich-powered full-screen display showing:
  - Summary statistics (total experiments, best CRPS, success rate)
  - Top experiments ranked by CRPS
  - Block and head performance breakdown
  - CRPS trend sparkline over time
  - Recent pipeline runs

Usage::

    synth-city display                  # full dashboard (Hippius data)
    synth-city display --limit 50       # limit experiments fetched
    synth-city display --source hippius  # explicit source selection
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.display import console, print_banner, section_header

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sparkline helpers
# ---------------------------------------------------------------------------

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 40) -> str:
    """Render a list of floats as a Unicode sparkline string.

    For CRPS (lower is better), the chart is inverted so that visual peaks
    represent *better* (lower) scores.
    """
    if not values:
        return ""
    lo, hi = min(values), max(values)
    rng = hi - lo if hi != lo else 1.0
    # Invert: lower CRPS -> taller bar
    chars = []
    for v in values:
        normalised = 1.0 - ((v - lo) / rng)  # 1.0 = best, 0.0 = worst
        idx = int(normalised * (len(_SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(_SPARK_CHARS) - 1))
        chars.append(_SPARK_CHARS[idx])
    # Downsample if wider than requested width
    if len(chars) > width:
        step = len(chars) / width
        chars = [chars[int(i * step)] for i in range(width)]
    return "".join(chars)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_hippius_data(limit: int) -> dict[str, Any]:
    """Load historical experiment data from Hippius storage."""
    try:
        from pipeline.tools.hippius_store import load_hippius_history
        raw = load_hippius_history(limit=limit)
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Failed to load Hippius data: %s", exc)
        return {"experiments": [], "total_stored": 0, "returned": 0}


def _load_scan_data(limit: int) -> dict[str, Any]:
    """Load the experiment scan analysis (block/head stats, lessons, etc.)."""
    try:
        from pipeline.tools.analysis_tools import _build_scan_result
        from pipeline.tools.hippius_store import _endpoint_unreachable, _get_json, _list_keys

        if _endpoint_unreachable:
            return {"total_experiments": 0}

        keys = _list_keys("experiments/", max_keys=2000)
        experiments: list[dict[str, Any]] = []
        consecutive_failures = 0
        for key in keys:
            if _endpoint_unreachable:
                break
            exp = _get_json(key)
            if exp is None:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    break
                continue
            consecutive_failures = 0
            if isinstance(exp, dict):
                experiments.append(exp)
            if len(experiments) >= limit:
                break

        return _build_scan_result(experiments)
    except Exception as exc:
        logger.warning("Failed to load scan data: %s", exc)
        return {"total_experiments": 0}


def _load_pipeline_runs() -> list[dict[str, Any]]:
    """Load the list of pipeline runs from Hippius."""
    try:
        from pipeline.tools.hippius_store import list_hippius_runs
        raw = list_hippius_runs()
        data = json.loads(raw)
        return data.get("runs", [])
    except Exception as exc:
        logger.warning("Failed to load pipeline runs: %s", exc)
        return []


def _load_trends(limit: int) -> dict[str, Any]:
    """Load CRPS trend data from Hippius."""
    try:
        from pipeline.tools.analysis_tools import analyze_experiment_trends
        raw = analyze_experiment_trends(limit=limit)
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Failed to load trends: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def _summary_panel(
    history: dict[str, Any],
    scan: dict[str, Any],
) -> Panel:
    """Build the overview summary panel."""
    total = scan.get("total_experiments", 0)
    successful = scan.get("successful", 0)
    failed = scan.get("failed", 0)
    rate = f"{successful / total * 100:.0f}%" if total else "--"

    top = scan.get("top_configs", [])
    best_crps = top[0]["crps"] if top else None
    best_name = top[0].get("name", "?") if top else "--"

    lines = Text()
    lines.append("  Total experiments:  ", style="bold white")
    lines.append(f"{total}\n", style="cyan")
    lines.append("  Successful:         ", style="bold white")
    lines.append(f"{successful}\n", style="bold green")
    lines.append("  Failed:             ", style="bold white")
    lines.append(f"{failed}\n", style="bold red")
    lines.append("  Success rate:       ", style="bold white")
    lines.append(f"{rate}\n", style="cyan")
    lines.append("  Duplicates found:   ", style="bold white")
    lines.append(f"{scan.get('duplicate_count', 0)}\n", style="dim")
    lines.append("\n")
    lines.append("  Best CRPS:          ", style="bold white")
    if best_crps is not None:
        lines.append(f"{best_crps:.6f}\n", style="bold magenta")
    else:
        lines.append("--\n", style="dim")
    lines.append("  Best experiment:    ", style="bold white")
    lines.append(f"{best_name}", style="cyan")

    stored = history.get("total_stored", "?")
    return Panel(
        lines,
        title=f"[bold cyan]Overview ({stored} stored)[/bold cyan]",
        border_style="cyan",
        padding=(0, 1),
    )


def _top_experiments_panel(scan: dict[str, Any]) -> Panel:
    """Build the top experiments ranking table."""
    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
        pad_edge=True,
        expand=True,
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Name", style="white", min_width=18, no_wrap=True)
    table.add_column("CRPS", style="bold magenta", justify="right", width=12)
    table.add_column("Blocks", style="dim", min_width=20)
    table.add_column("Head", style="cyan", width=16)
    table.add_column("d_model", style="dim", justify="right", width=8)
    table.add_column("lr", style="dim", justify="right", width=10)

    top = scan.get("top_configs", [])
    for i, cfg in enumerate(top, 1):
        crps = cfg.get("crps")
        crps_str = f"{crps:.6f}" if isinstance(crps, (int, float)) else "--"
        blocks = ", ".join(cfg.get("blocks", []))
        table.add_row(
            str(i),
            str(cfg.get("name", "?")),
            crps_str,
            blocks or "--",
            str(cfg.get("head", "?")),
            str(cfg.get("d_model", "?")),
            str(cfg.get("lr", "?")),
        )

    if not top:
        table.add_row("", "[dim]No successful experiments yet[/dim]", "", "", "", "", "")

    return Panel(
        table,
        title=f"[bold cyan]Top Experiments ({len(top)})[/bold cyan]",
        border_style="cyan",
    )


def _block_stats_panel(scan: dict[str, Any]) -> Panel:
    """Build the block performance table."""
    table = Table(
        show_header=True,
        header_style="bold green",
        border_style="green",
        pad_edge=True,
        expand=True,
    )
    table.add_column("Block", style="white", min_width=16)
    table.add_column("Best CRPS", style="bold magenta", justify="right", width=12)
    table.add_column("Mean CRPS", style="magenta", justify="right", width=12)
    table.add_column("Uses", style="dim", justify="right", width=6)

    block_stats = scan.get("block_stats", {})
    for block, stats in block_stats.items():
        best = stats.get("best_crps")
        mean = stats.get("mean_crps")
        table.add_row(
            block,
            f"{best:.6f}" if best is not None else "--",
            f"{mean:.6f}" if mean is not None else "--",
            str(stats.get("count", 0)),
        )

    if not block_stats:
        table.add_row("[dim]No data[/dim]", "", "", "")

    untried = scan.get("untried_blocks", [])
    subtitle = f"[dim]{len(untried)} untried[/dim]" if untried else ""
    return Panel(
        table,
        title="[bold green]Block Performance[/bold green]",
        subtitle=subtitle,
        border_style="green",
    )


def _head_stats_panel(scan: dict[str, Any]) -> Panel:
    """Build the head performance table."""
    table = Table(
        show_header=True,
        header_style="bold yellow",
        border_style="yellow",
        pad_edge=True,
        expand=True,
    )
    table.add_column("Head", style="white", min_width=16)
    table.add_column("Best CRPS", style="bold magenta", justify="right", width=12)
    table.add_column("Mean CRPS", style="magenta", justify="right", width=12)
    table.add_column("Uses", style="dim", justify="right", width=6)

    head_stats = scan.get("head_stats", {})
    for head, stats in head_stats.items():
        best = stats.get("best_crps")
        mean = stats.get("mean_crps")
        table.add_row(
            head,
            f"{best:.6f}" if best is not None else "--",
            f"{mean:.6f}" if mean is not None else "--",
            str(stats.get("count", 0)),
        )

    if not head_stats:
        table.add_row("[dim]No data[/dim]", "", "", "")

    untried = scan.get("untried_heads", [])
    subtitle = f"[dim]{len(untried)} untried[/dim]" if untried else ""
    return Panel(
        table,
        title="[bold yellow]Head Performance[/bold yellow]",
        subtitle=subtitle,
        border_style="yellow",
    )


def _trend_panel(trends: dict[str, Any]) -> Panel:
    """Build the CRPS trend panel with sparkline."""
    timeline = trends.get("timeline", [])
    crps_values = [e["crps"] for e in timeline if isinstance(e.get("crps"), (int, float))]

    lines = Text()

    if crps_values:
        spark = _sparkline(crps_values, width=60)
        lines.append("  CRPS over time (taller = better):\n\n", style="dim")
        lines.append(f"  {spark}\n\n", style="bold cyan")

        lines.append("  Total runs:     ", style="bold white")
        lines.append(f"{trends.get('total_runs', '?')}\n", style="cyan")
        lines.append("  Best CRPS:      ", style="bold white")
        lines.append(f"{trends.get('best_crps', 'N/A')}\n", style="bold green")
        lines.append("  Best run:       ", style="bold white")
        lines.append(f"{trends.get('best_run', 'N/A')}\n", style="cyan")
        lines.append("  Latest CRPS:    ", style="bold white")
        lines.append(f"{trends.get('latest_crps', 'N/A')}\n", style="cyan")

        improvement = trends.get("improvement", 0)
        lines.append("  Improvement:    ", style="bold white")
        if isinstance(improvement, (int, float)) and improvement > 0:
            lines.append(f"-{improvement:.6f}", style="bold green")
        elif isinstance(improvement, (int, float)):
            lines.append(f"{improvement:.6f}", style="dim")
        else:
            lines.append("--", style="dim")
    else:
        lines.append("  No trend data available yet.", style="dim")

    return Panel(
        lines,
        title="[bold cyan]CRPS Trend[/bold cyan]",
        border_style="cyan",
    )


def _pipeline_runs_panel(runs: list[dict[str, Any]]) -> Panel:
    """Build the recent pipeline runs panel."""
    table = Table(
        show_header=True,
        header_style="bold magenta",
        border_style="magenta",
        pad_edge=True,
        expand=True,
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Run ID", style="cyan", min_width=24)
    table.add_column("Files", style="dim", width=30)

    for i, run in enumerate(runs[:10], 1):
        run_id = run.get("run_id", "?")
        files = ", ".join(run.get("files", []))
        table.add_row(str(i), run_id, files)

    if not runs:
        table.add_row("", "[dim]No pipeline runs found[/dim]", "")

    return Panel(
        table,
        title=f"[bold magenta]Pipeline Runs ({len(runs)} total)[/bold magenta]",
        border_style="magenta",
    )


def _lessons_panel(scan: dict[str, Any]) -> Panel:
    """Build the lessons-learned panel."""
    lessons = scan.get("lessons", "")
    if not lessons or lessons == "Insufficient data.":
        content = Text("  No lessons yet — run some experiments first.", style="dim")
    else:
        content = Text()
        for line in lessons.split("\n"):
            if line.startswith("- "):
                content.append("  ", style="dim")
                content.append(line[2:], style="white")
                content.append("\n")
            else:
                content.append(f"  {line}\n", style="white")

    return Panel(
        content,
        title="[bold cyan]Lessons Learned[/bold cyan]",
        border_style="cyan",
    )


def _untried_panel(scan: dict[str, Any]) -> Panel:
    """Build the untried combinations panel."""
    lines = Text()

    untried_blocks = scan.get("untried_blocks", [])
    untried_heads = scan.get("untried_heads", [])
    untried_pairs = scan.get("untried_block_head_pairs", [])

    if untried_blocks:
        lines.append("  Untried blocks: ", style="bold white")
        lines.append(", ".join(untried_blocks), style="yellow")
        lines.append("\n")
    if untried_heads:
        lines.append("  Untried heads:  ", style="bold white")
        lines.append(", ".join(untried_heads), style="yellow")
        lines.append("\n")

    if untried_pairs:
        lines.append("\n  Untried block+head combos:\n", style="bold white")
        for pair in untried_pairs[:8]:
            lines.append(f"    {pair['block']} + {pair['head']}\n", style="dim")
        remaining = len(untried_pairs) - 8
        if remaining > 0:
            lines.append(f"    ... and {remaining} more\n", style="dim")

    if not untried_blocks and not untried_heads and not untried_pairs:
        lines.append("  All known components have been tried!", style="bold green")

    return Panel(
        lines,
        title="[bold yellow]Exploration Frontier[/bold yellow]",
        border_style="yellow",
    )


def _failure_panel(scan: dict[str, Any]) -> Panel:
    """Build the failure patterns panel."""
    patterns = scan.get("failure_patterns", [])
    lines = Text()

    if patterns:
        for p in patterns[:6]:
            count = p.get("count", 0)
            error = p.get("error", "?")
            lines.append(f"  {count:>3}x  ", style="bold red")
            lines.append(f"{error}\n", style="white")
    else:
        lines.append("  No failures recorded.", style="dim")

    return Panel(
        lines,
        title="[bold red]Common Failures[/bold red]",
        border_style="red",
    )


# ---------------------------------------------------------------------------
# Full history table (detailed)
# ---------------------------------------------------------------------------

def _full_history_table(experiments: list[dict[str, Any]]) -> Panel:
    """Build a detailed history table of all fetched experiments."""
    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
        pad_edge=True,
        expand=True,
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Run ID", style="dim", width=16)
    table.add_column("Name", style="white", min_width=18, no_wrap=True)
    table.add_column("CRPS", style="bold magenta", justify="right", width=12)
    table.add_column("Sharpness", style="dim", justify="right", width=12)
    table.add_column("LogLik", style="dim", justify="right", width=12)
    table.add_column("Blocks", style="dim", min_width=18)
    table.add_column("Head", style="cyan", width=14)

    for i, exp in enumerate(experiments, 1):
        crps = exp.get("crps")
        metrics = exp.get("metrics", {})
        sharpness = metrics.get("sharpness")
        loglik = metrics.get("log_likelihood")
        blocks = ", ".join(exp.get("blocks", []))

        table.add_row(
            str(i),
            str(exp.get("run_id", "?"))[:15],
            str(exp.get("name", "?")),
            f"{crps:.6f}" if isinstance(crps, (int, float)) else "--",
            f"{sharpness:.6f}" if isinstance(sharpness, (int, float)) else "--",
            f"{loglik:.4f}" if isinstance(loglik, (int, float)) else "--",
            blocks or "--",
            str(exp.get("head", "?")),
        )

    if not experiments:
        table.add_row("", "", "[dim]No experiments found[/dim]", "", "", "", "", "")

    return Panel(
        table,
        title=f"[bold cyan]Experiment History ({len(experiments)} shown)[/bold cyan]",
        border_style="cyan",
    )


# ---------------------------------------------------------------------------
# Main display entry point
# ---------------------------------------------------------------------------

def run_display(limit: int = 50) -> None:
    """Fetch historical data and render the full display dashboard."""
    print_banner(subtitle="historical results")
    console.print()

    # Load data with progress indication
    with console.status("[bold cyan]Loading experiment history from Hippius..."):
        history = _load_hippius_data(limit=limit)
        scan = _load_scan_data(limit=limit)
        trends = _load_trends(limit=limit)
        runs = _load_pipeline_runs()

    experiments = history.get("experiments", [])
    has_data = len(experiments) > 0 or scan.get("total_experiments", 0) > 0

    if not has_data:
        section_header("No Data")
        console.print(
            "[dim]  No historical experiments found in Hippius storage.\n"
            "  Run the pipeline first:[/dim]\n\n"
            "    [bold cyan]synth-city pipeline[/bold cyan]\n\n"
            "[dim]  Or check your Hippius configuration in .env[/dim]\n"
        )
        return

    # ── Summary + Trend ──────────────────────────────────────────────
    section_header("Summary")
    console.print(Columns([
        _summary_panel(history, scan),
        _trend_panel(trends),
    ], equal=True, expand=True))
    console.print()

    # ── Top Experiments ──────────────────────────────────────────────
    section_header("Top Experiments")
    console.print(_top_experiments_panel(scan))
    console.print()

    # ── Block & Head Performance ─────────────────────────────────────
    section_header("Component Performance")
    console.print(Columns([
        _block_stats_panel(scan),
        _head_stats_panel(scan),
    ], equal=True, expand=True))
    console.print()

    # ── Exploration + Failures ───────────────────────────────────────
    section_header("Insights")
    console.print(Columns([
        _untried_panel(scan),
        _failure_panel(scan),
    ], equal=True, expand=True))
    console.print()

    # ── Lessons ──────────────────────────────────────────────────────
    console.print(_lessons_panel(scan))
    console.print()

    # ── Full History Table ───────────────────────────────────────────
    section_header("Experiment History")
    console.print(_full_history_table(experiments))
    console.print()

    # ── Pipeline Runs ────────────────────────────────────────────────
    if runs:
        section_header("Pipeline Runs")
        console.print(_pipeline_runs_panel(runs))
        console.print()
