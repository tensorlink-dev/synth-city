"""CLI display helpers — Rich-powered output for synth-city."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ── Theme ────────────────────────────────────────────────────────────────────

_THEME = Theme({
    "header": "bold cyan",
    "success": "bold green",
    "error": "bold red",
    "warning": "bold yellow",
    "muted": "dim",
    "metric": "bold magenta",
    "label": "bold white",
    "accent": "cyan",
})

console = Console(theme=_THEME, highlight=False)

# ── Banner ───────────────────────────────────────────────────────────────────

_BANNER_LINES = [
    "███████╗██╗   ██╗███╗   ██╗████████╗██╗  ██╗     ██████╗██╗████████╗██╗   ██╗",
    "██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██║  ██║    ██╔════╝██║╚══██╔══╝╚██╗ ██╔╝",
    "███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ███████║    ██║     ██║   ██║    ╚████╔╝",
    "╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██║    ██║     ██║   ██║     ╚██╔╝",
    "███████║   ██║   ██║ ╚████║   ██║   ██║  ██║    ╚██████╗██║   ██║      ██║",
    "╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝     ╚═════╝╚═╝   ╚═╝      ╚═╝",
]

_GRADIENT = ["#00d4ff", "#00b4ff", "#0094ff", "#0074ff", "#0054ff", "#0034ff"]


def print_banner(subtitle: str = "agentic pipeline for Bittensor SN50") -> None:
    """Print the stylised SYNTH CITY banner with gradient colouring."""
    text = Text()
    for i, line in enumerate(_BANNER_LINES):
        colour = _GRADIENT[i % len(_GRADIENT)]
        text.append(line, style=colour)
        if i < len(_BANNER_LINES) - 1:
            text.append("\n")

    panel = Panel(
        text,
        subtitle=f"[dim]{subtitle}[/dim]",
        border_style="bright_cyan",
        padding=(1, 2),
    )
    console.print(panel)


# ── Tables ───────────────────────────────────────────────────────────────────


def ranking_table(
    ranking: list[dict[str, Any]],
    title: str = "Ranking (by CRPS, best first)",
) -> None:
    """Print a styled ranking table (sweep / compare results)."""
    table = Table(title=title, border_style="cyan", header_style="bold cyan", pad_edge=True)
    table.add_column("#", style="muted", justify="right", width=4)
    table.add_column("Name", style="label", min_width=20)
    table.add_column("CRPS", style="metric", justify="right")
    table.add_column("Params", style="accent", justify="right")

    for i, entry in enumerate(ranking, 1):
        crps = entry.get("crps", "N/A")
        crps_str = f"{crps:.6f}" if isinstance(crps, (int, float)) else str(crps)
        table.add_row(
            str(i),
            str(entry.get("name", "?")),
            crps_str,
            str(entry.get("param_count", "?")),
        )
    console.print(table)


def hippius_table(experiments: list[dict[str, Any]], total: int | str = "?") -> None:
    """Print a styled Hippius history table."""
    table = Table(
        title=f"Hippius History ({total} total)",
        border_style="cyan",
        header_style="bold cyan",
    )
    table.add_column("#", style="muted", justify="right", width=4)
    table.add_column("Run ID", style="accent", width=16)
    table.add_column("Name", style="label", min_width=20)
    table.add_column("CRPS", style="metric", justify="right")
    table.add_column("Blocks", style="dim")
    table.add_column("Head", style="dim")

    for i, exp in enumerate(experiments, 1):
        crps = exp.get("crps", "N/A")
        crps_str = f"{crps:.6f}" if isinstance(crps, (int, float)) else str(crps)
        blocks = ", ".join(exp.get("blocks", []))
        table.add_row(
            str(i),
            str(exp.get("run_id", "?"))[:15],
            str(exp.get("name", "?")),
            crps_str,
            blocks,
            str(exp.get("head", "?")),
        )
    console.print(table)


def wandb_runs_table(runs: list[dict[str, Any]], order: str = "best") -> None:
    """Print a styled W&B runs table."""
    table = Table(
        title=f"W&B Runs (order={order})",
        border_style="cyan",
        header_style="bold cyan",
    )
    table.add_column("#", style="muted", justify="right", width=4)
    table.add_column("Name", style="label", min_width=30)
    table.add_column("CRPS", style="metric", justify="right")
    table.add_column("State", justify="center")

    for i, run in enumerate(runs, 1):
        crps = run.get("crps", "N/A")
        crps_str = f"{crps:.6f}" if isinstance(crps, (int, float)) else str(crps)
        state = str(run.get("state", "?"))
        state_style = "green" if state == "finished" else "yellow" if state == "running" else "dim"
        table.add_row(
            str(i),
            str(run.get("name", "?")),
            crps_str,
            f"[{state_style}]{state}[/{state_style}]",
        )
    console.print(table)


def wandb_trends_panel(data: dict[str, Any]) -> None:
    """Print a styled W&B CRPS trends panel."""
    lines = Text()
    lines.append("  Best CRPS:   ", style="label")
    lines.append(f"{data.get('best_crps', 'N/A')}\n", style="metric")
    lines.append("  Best run:    ", style="label")
    lines.append(f"{data.get('best_run', 'N/A')}\n", style="accent")
    lines.append("  Latest CRPS: ", style="label")
    lines.append(f"{data.get('latest_crps', 'N/A')}\n", style="metric")
    lines.append("  Improvement: ", style="label")
    improvement = data.get("improvement", "N/A")
    imp_style = "success" if str(improvement).startswith("-") else "warning"
    lines.append(f"{improvement}", style=imp_style)

    panel = Panel(
        lines,
        title=f"[bold cyan]W&B CRPS Trends ({data.get('total_runs', '?')} runs)[/bold cyan]",
        border_style="cyan",
    )
    console.print(panel)


def hf_panel(data: dict[str, Any]) -> None:
    """Print a styled HF Hub info panel."""
    lines = Text()
    lines.append("  Downloads: ", style="label")
    lines.append(f"{data.get('downloads', 'N/A')}\n", style="accent")
    lines.append("  Likes:     ", style="label")
    lines.append(f"{data.get('likes', 'N/A')}\n", style="accent")
    lines.append("  Files:     ", style="label")
    lines.append(f"{len(data.get('files', []))}\n", style="accent")

    for f in data.get("files", []):
        lines.append(f"    {f.get('path', '?')}\n", style="muted")

    panel = Panel(
        lines,
        title=f"[bold cyan]HF Hub: {data.get('repo_id', '?')}[/bold cyan]",
        border_style="cyan",
    )
    console.print(panel)


# ── Metrics / results ────────────────────────────────────────────────────────


def metrics_panel(metrics: dict[str, Any], title: str = "Experiment Results") -> None:
    """Display key metrics in a styled panel."""
    table = Table(show_header=False, border_style="cyan", pad_edge=True, box=None)
    table.add_column("Metric", style="label", min_width=16)
    table.add_column("Value", style="metric")

    for key, val in metrics.items():
        val_str = f"{val:.6f}" if isinstance(val, float) else str(val)
        table.add_row(key, val_str)

    panel = Panel(table, title=f"[bold cyan]{title}[/bold cyan]", border_style="cyan")
    console.print(panel)


def agent_result_panel(
    name: str, success: bool, turns: int,
    structured: dict | None = None, raw_text: str | None = None,
) -> None:
    """Display agent run result in a styled panel."""
    status = "[success]SUCCESS[/success]" if success else "[error]FAILED[/error]"

    lines = Text()
    lines.append("  Agent:   ", style="label")
    lines.append(f"{name}\n", style="accent")
    lines.append("  Status:  ", style="label")
    console.print(
        Panel(
            lines,
            title="[bold cyan]Agent Result[/bold cyan]",
            subtitle=f"{status}  [muted]({turns} turns)[/muted]",
            border_style="green" if success else "red",
        )
    )

    if structured:
        console.print(
            Panel(
                JSON.from_data(structured),
                title="[bold cyan]Structured Output[/bold cyan]",
                border_style="cyan",
            )
        )
    if raw_text:
        console.print(
            Panel(
                raw_text[:2000],
                title="[bold cyan]Raw Text[/bold cyan]",
                border_style="dim",
            )
        )


# ── JSON output ──────────────────────────────────────────────────────────────


def print_json(data: Any) -> None:
    """Print syntax-highlighted JSON."""
    console.print(JSON.from_data(data))


# ── Status helpers ───────────────────────────────────────────────────────────


def print_success(msg: str) -> None:
    console.print(f"[success]  {msg}[/success]")


def print_error(msg: str) -> None:
    console.print(f"[error]  {msg}[/error]")


def print_warning(msg: str) -> None:
    console.print(f"[warning]  {msg}[/warning]")


def section_header(title: str) -> None:
    """Print a section separator."""
    console.rule(f"[bold cyan]{title}[/bold cyan]", style="bright_cyan")


def print_validation_errors(errors: list[str]) -> None:
    """Print validation errors as a styled list."""
    console.print("[error]Validation FAILED[/error]")
    for err in errors:
        console.print(f"  [error]  {err}[/error]")
