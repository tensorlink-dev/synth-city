"""
synth-city CLI application.

Usage:
    synth-city pipeline                 # Run the full improvement pipeline
    synth-city sweep                    # Run a preset sweep
    synth-city experiment --blocks ...  # Run a single experiment
    synth-city quick --blocks ...       # One-liner convenience experiment
    synth-city history hippius          # Query experiment history
    synth-city bridge                   # Start the HTTP bridge server
    synth-city client blocks            # Talk to a running bridge
    synth-city agent --name planner     # Run a single agent
    synth-city data download            # Pre-download HF training data
    synth-city data info                # Show data loader configuration
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from cli.display import (
    agent_result_panel,
    console,
    experiment_runs_table,
    experiment_trends_panel,
    hf_panel,
    hippius_table,
    metrics_panel,
    print_banner,
    print_error,
    print_json,
    print_validation_errors,
    ranking_table,
    section_header,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("synth-city")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Run the full agentic improvement pipeline."""
    from pipeline.bootstrap import bootstrap_all
    from pipeline.orchestrator import PipelineOrchestrator

    bootstrap_all()
    section_header("Pipeline")
    task: dict = {"channel": args.channel}
    orchestrator = PipelineOrchestrator(
        max_retries=args.retries,
        base_temperature=args.temperature,
        publish=args.publish,
    )
    result = orchestrator.run(task)
    print_json(result)


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run a quick preset sweep via ResearchSession."""
    from pipeline.bootstrap import bootstrap_dirs
    from pipeline.tools.research_tools import _import_research_session

    bootstrap_dirs()
    section_header("Sweep")
    ResearchSession = _import_research_session()
    session = ResearchSession()
    presets = [p.strip() for p in args.presets.split(",")] if args.presets else None
    result = session.sweep(preset_names=presets, epochs=args.epochs)

    print_json(result)

    comparison = session.compare()
    ranking = comparison.get("ranking", [])
    if ranking:
        console.print()
        ranking_table(ranking)


def cmd_experiment(args: argparse.Namespace) -> None:
    """Run a single experiment."""
    from pipeline.bootstrap import bootstrap_dirs
    from pipeline.tools.research_tools import _import_research_session

    bootstrap_dirs()
    section_header("Experiment")
    ResearchSession = _import_research_session()
    session = ResearchSession()
    blocks = [b.strip() for b in args.blocks.split(",")]

    experiment = session.create_experiment(
        blocks=blocks,
        head=args.head,
        d_model=args.d_model,
        horizon=args.horizon,
        n_paths=args.n_paths,
        lr=args.lr,
    )

    # Validate first
    validation = session.validate(experiment)
    if not validation["valid"]:
        print_validation_errors(validation["errors"])
        sys.exit(1)

    # Run
    result = session.run(experiment, epochs=args.epochs)
    print_json(result)

    if result.get("status") == "ok":
        metrics_panel(result["metrics"])


def cmd_quick(args: argparse.Namespace) -> None:
    """One-liner convenience experiment."""
    import importlib

    from pipeline.bootstrap import bootstrap_dirs

    bootstrap_dirs()
    section_header("Quick Experiment")
    errors: list[tuple[str, Exception]] = []
    for _mod_path in ("osa.research.agent_api", "src.research.agent_api", "research.agent_api"):
        try:
            _mod = importlib.import_module(_mod_path)
            quick_experiment = getattr(_mod, "quick_experiment", None)
            if quick_experiment is not None:
                break
        except Exception as exc:
            errors.append((_mod_path, exc))
    else:
        details = "; ".join(f"{p}: {type(e).__name__}: {e}" for p, e in errors)
        raise ImportError(
            f"Cannot import quick_experiment from any known module path. "
            f"Errors: {details}"
        )
    blocks = [b.strip() for b in args.blocks.split(",")] if args.blocks else None
    result = quick_experiment(
        blocks=blocks,
        head=args.head,
        d_model=args.d_model,
        horizon=args.horizon,
    )
    print_json(result)


def cmd_bridge(args: argparse.Namespace) -> None:
    """Start the HTTP bridge server."""
    from integrations.openclaw.bridge import run_bridge

    run_bridge(host=args.host, port=args.port)


def cmd_client(args: argparse.Namespace) -> None:
    """CLI client for the bridge server — no OpenClaw required."""
    from integrations.openclaw.client import SynthCityClient

    client = SynthCityClient(base_url=f"http://{args.host}:{args.port}")
    action = args.action

    try:
        if action == "health":
            result = client.health()
        elif action == "blocks":
            result = client.list_blocks()
        elif action == "heads":
            result = client.list_heads()
        elif action == "presets":
            result = client.list_presets()
        elif action == "compare":
            result = client.compare_results()
        elif action == "summary":
            result = client.session_summary()
        elif action == "clear":
            result = client.clear_session()
        elif action == "status":
            result = client.pipeline_status()
        elif action == "run":
            result = client.pipeline_run(publish=args.publish)
        elif action == "price":
            if not args.extra:
                print_error("Usage: synth-city client price <ASSET>")
                sys.exit(1)
            result = client.get_price(args.extra[0])
        elif action == "history":
            if not args.extra:
                print_error("Usage: synth-city client history <ASSET> [days]")
                sys.exit(1)
            days = int(args.extra[1]) if len(args.extra) > 1 else 30
            result = client.get_history(args.extra[0], days=days)
        else:
            print_error(f"Unknown action: {action}")
            console.print(
                "[muted]Available: health blocks heads presets compare "
                "summary clear status run price history[/muted]"
            )
            sys.exit(1)

        print_json(result)
    except Exception as exc:
        print_error(f"Error: {exc}")
        console.print(
            "[muted]Is the bridge running? Start it with: synth-city bridge[/muted]"
        )
        sys.exit(1)


def cmd_history(args: argparse.Namespace) -> None:
    """Query experiment history from Hippius, Trackio, or HF Hub."""
    source = args.source

    if source == "hippius":
        from pipeline.tools.hippius_store import load_hippius_history, load_hippius_run

        if args.run_id:
            result = load_hippius_run(args.run_id)
        else:
            result = load_hippius_history(limit=args.limit)
        data = json.loads(result)
        print_json(data)

        if not args.run_id and "experiments" in data:
            console.print()
            hippius_table(data["experiments"], total=data.get("total_stored", "?"))

    elif source == "trackio":
        from pipeline.tools.analysis_tools import (
            analyze_experiment_trends,
            fetch_experiment_runs,
        )

        if args.trends:
            result = analyze_experiment_trends(limit=args.limit)
            data = json.loads(result)
            print_json(data)
            if "timeline" in data:
                console.print()
                experiment_trends_panel(data)
        else:
            result = fetch_experiment_runs(limit=args.limit, order=args.order)
            data = json.loads(result)
            print_json(data)
            if "runs" in data:
                console.print()
                experiment_runs_table(data["runs"], order=args.order)

    elif source == "hf":
        from pipeline.tools.analysis_tools import list_hf_models

        result = list_hf_models(repo_id=args.repo_id or "")
        data = json.loads(result)
        print_json(data)
        if "files" in data:
            console.print()
            hf_panel(data)

    else:
        print_error(f"Unknown source: {source}")
        console.print("[muted]Available: hippius, trackio, hf[/muted]")
        sys.exit(1)


def cmd_data(args: argparse.Namespace) -> None:
    """Download or inspect HuggingFace training data."""
    from config import SN50_TO_HF_ASSET, TIMEFRAME_CONFIGS
    from pipeline.tools.data_loader import data_loader_info, get_loader

    action = args.action

    if action == "info":
        result = data_loader_info()
        print_json(json.loads(result))
        return

    # action == "download"
    asset_filter: list[str] | None = None
    if args.assets:
        asset_filter = [a.strip().upper() for a in args.assets.split(",")]
        unknown = [a for a in asset_filter if a not in SN50_TO_HF_ASSET]
        if unknown:
            print_error(
                f"Unknown asset(s): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(SN50_TO_HF_ASSET.keys()))}"
            )
            import sys
            sys.exit(1)

    timeframes: list[str]
    if args.timeframe == "all":
        timeframes = list(TIMEFRAME_CONFIGS.keys())
    else:
        timeframes = [args.timeframe]

    section_header("Data Download")
    asset_label = ", ".join(asset_filter) if asset_filter else "all"
    console.print(
        f"[bold]Assets:[/bold] {asset_label}  "
        f"[bold]Timeframes:[/bold] {', '.join(timeframes)}"
    )
    console.print()

    for tf in timeframes:
        console.print(f"[cyan]Downloading {tf} data…[/cyan]")
        try:
            loader = get_loader(timeframe=tf, assets=asset_filter, force_new=True)
            # Trigger actual data loading by peeking at the loader
            try:
                asset_names = [
                    a.name if hasattr(a, "name") else str(a)
                    for a in loader.assets_data
                ]
            except Exception:
                asset_names = asset_filter or list(SN50_TO_HF_ASSET.values())
            console.print(
                f"  [green]✓[/green] {tf}: {len(asset_names)} asset(s) cached "
                f"— {', '.join(asset_names)}"
            )
        except Exception as exc:
            print_error(f"  {tf}: {exc}")

    console.print()
    console.print("[bold green]Done.[/bold green] Data is cached for future pipeline runs.")


def cmd_score(args: argparse.Namespace) -> None:
    """Scoring emulator — local replica of SN50 validator scoring."""
    from cli.score_dashboard import (
        print_daily_summary,
        print_leaderboard,
        print_prompt_results,
        print_score_status,
        run_score_dashboard,
    )

    action = args.action

    if action == "status":
        from subnet.score_tracker import ScoreTracker
        tracker = ScoreTracker()
        print_score_status(tracker.get_status())

    elif action == "results":
        from subnet.score_tracker import ScoreTracker
        tracker = ScoreTracker()
        date = getattr(args, "date", None)
        results = tracker.load_prompt_history(date_str=date, limit=args.limit)
        print_prompt_results(results, date=date)

    elif action == "daily":
        from subnet.score_tracker import ScoreTracker
        tracker = ScoreTracker()
        date = getattr(args, "date", None)
        summary = tracker.save_daily_summary(date_str=date)
        print_daily_summary(summary)

    elif action == "leaderboard":
        from subnet.score_tracker import ScoreTracker
        tracker = ScoreTracker()
        lb = tracker.load_leaderboard()
        print_leaderboard(lb)

    elif action == "run":
        interval = getattr(args, "interval", 5)
        run_score_dashboard(interval_minutes=interval)

    else:
        print_error(f"Unknown action: {action}")
        console.print("[muted]Available: run, status, results, daily, leaderboard[/muted]")
        sys.exit(1)


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Run the pipeline with a live Rich dashboard, or monitor a remote bridge."""
    if args.remote:
        from cli.dashboard import run_dashboard_remote

        print_banner(subtitle="remote dashboard")
        run_dashboard_remote(args.remote)
    else:
        from cli.dashboard import run_dashboard

        print_banner(subtitle="dashboard")
        task: dict = {"channel": args.channel}
        result = run_dashboard(
            task,
            max_retries=args.retries,
            base_temperature=args.temperature,
            publish=args.publish,
        )
        section_header("Pipeline Result")
        print_json(result)


def cmd_agent(args: argparse.Namespace) -> None:
    """Run a single agent for debugging/testing."""
    from pipeline.bootstrap import bootstrap_dirs

    bootstrap_dirs()

    from pipeline.agents.agent_designer import AgentDesignerAgent
    from pipeline.agents.author import ComponentAuthorAgent
    from pipeline.agents.code_checker import CodeCheckerAgent
    from pipeline.agents.debugger import DebuggerAgent
    from pipeline.agents.planner import PlannerAgent
    from pipeline.agents.publisher import PublisherAgent
    from pipeline.agents.trainer import TrainerAgent

    agents = {
        "planner": PlannerAgent,
        "codechecker": CodeCheckerAgent,
        "debugger": DebuggerAgent,
        "trainer": TrainerAgent,
        "publisher": PublisherAgent,
        "author": ComponentAuthorAgent,
        "agent_designer": AgentDesignerAgent,
    }

    agent_cls = agents.get(args.name.lower())
    if not agent_cls:
        print_error(f"Unknown agent: {args.name}")
        console.print(f"[muted]Available: {', '.join(agents.keys())}[/muted]")
        sys.exit(1)

    section_header(f"Agent: {args.name}")
    task: dict = {
        "channel": "default",
        "user_message": args.message or "Begin the task.",
    }

    agent = agent_cls(temperature=args.temperature)
    result = agent.run(task)
    agent_result_panel(
        name=args.name,
        success=result.success,
        turns=result.turns_used,
        structured=result.structured,
        raw_text=result.raw_text,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="synth-city",
        description="synth-city — agentic pipeline for Bittensor SN50 competition",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Run the full improvement pipeline")
    p_pipe.add_argument("--channel", default="default", help="Prompt channel")
    p_pipe.add_argument("--retries", type=int, default=5, help="Max retries per stage")
    p_pipe.add_argument("--temperature", type=float, default=0.1, help="Base LLM temperature")
    p_pipe.add_argument("--publish", action="store_true", help="Publish best model to HF Hub")

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Run a preset sweep")
    p_sweep.add_argument(
        "--presets", default=None, help="Comma-separated preset names (all if omitted)"
    )
    p_sweep.add_argument("--epochs", type=int, default=1, help="Epochs per preset")

    # experiment
    p_exp = subparsers.add_parser("experiment", help="Run a single experiment")
    p_exp.add_argument("--blocks", required=True, help="Comma-separated block names")
    p_exp.add_argument("--head", default="GBMHead", help="Head name")
    p_exp.add_argument("--d-model", type=int, default=32, help="Hidden dimension")
    p_exp.add_argument("--horizon", type=int, default=12, help="Prediction steps")
    p_exp.add_argument("--n-paths", type=int, default=100, help="Monte Carlo paths")
    p_exp.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    p_exp.add_argument("--epochs", type=int, default=1, help="Training epochs")

    # quick
    p_quick = subparsers.add_parser("quick", help="One-liner convenience experiment")
    p_quick.add_argument("--blocks", default=None, help="Comma-separated block names")
    p_quick.add_argument("--head", default="GBMHead", help="Head name")
    p_quick.add_argument("--d-model", type=int, default=32, help="Hidden dimension")
    p_quick.add_argument("--horizon", type=int, default=12, help="Prediction steps")

    # bridge
    from config import BRIDGE_HOST, BRIDGE_PORT

    p_bridge = subparsers.add_parser("bridge", help="Start the HTTP bridge server")
    p_bridge.add_argument("--host", default=BRIDGE_HOST, help="Bind address")
    p_bridge.add_argument("--port", type=int, default=BRIDGE_PORT, help="Listen port")

    # client (standalone CLI for the bridge)
    p_client = subparsers.add_parser(
        "client",
        help="Talk to a running bridge server (no OpenClaw needed)",
    )
    p_client.add_argument(
        "action",
        choices=[
            "health", "blocks", "heads", "presets", "compare", "summary",
            "clear", "status", "run", "price", "history",
        ],
        help="Action to perform",
    )
    p_client.add_argument(
        "extra", nargs="*", help="Extra args (e.g. asset name for price/history)"
    )
    p_client.add_argument("--host", default=BRIDGE_HOST, help="Bridge host")
    p_client.add_argument("--port", type=int, default=BRIDGE_PORT, help="Bridge port")
    p_client.add_argument(
        "--publish", action="store_true", help="Publish when using 'run' action"
    )

    # history
    p_hist = subparsers.add_parser(
        "history", help="Query experiment history (Hippius, Trackio, HF Hub)"
    )
    p_hist.add_argument("source", choices=["hippius", "trackio", "hf"], help="Data source")
    p_hist.add_argument("--limit", type=int, default=20, help="Max results to return")
    p_hist.add_argument(
        "--order", default="best", choices=["best", "recent", "worst"],
        help="Sort order for experiment runs",
    )
    p_hist.add_argument("--trends", action="store_true", help="Show CRPS trends (trackio only)")
    p_hist.add_argument(
        "--run-id", default=None,
        help="Load a specific Hippius run ID ('latest' for most recent)",
    )
    p_hist.add_argument("--repo-id", default=None, help="HF Hub repo ID override")

    # data
    p_data = subparsers.add_parser(
        "data", help="Download or inspect HuggingFace training data"
    )
    p_data.add_argument(
        "action",
        choices=["download", "info"],
        help="'download' to pre-fetch data, 'info' to show config and active loaders",
    )
    p_data.add_argument(
        "--assets", default=None,
        help="Comma-separated SN50 asset names to download (default: all). "
             "E.g. --assets BTC or --assets BTC,ETH,SOL",
    )
    p_data.add_argument(
        "--timeframe", default="5m", choices=["5m", "1m", "all"],
        help="Candle timeframe to download: '5m', '1m', or 'all' (default: 5m)",
    )

    # score
    p_score = subparsers.add_parser(
        "score", help="Scoring emulator — local replica of SN50 validator scoring"
    )
    p_score.add_argument(
        "action",
        choices=["run", "status", "results", "daily", "leaderboard"],
        help="Action: run (start daemon with dashboard), status, results, daily, leaderboard",
    )
    p_score.add_argument("--interval", type=int, default=5, help="Prompt interval in minutes")
    p_score.add_argument("--date", default=None, help="Date filter (YYYY-MM-DD)")
    p_score.add_argument("--limit", type=int, default=20, help="Max results to return")

    # dashboard
    p_dash = subparsers.add_parser(
        "dashboard", help="Run pipeline with live monitoring dashboard"
    )
    p_dash.add_argument("--channel", default="default", help="Prompt channel")
    p_dash.add_argument("--retries", type=int, default=5, help="Max retries per stage")
    p_dash.add_argument("--temperature", type=float, default=0.1, help="Base LLM temperature")
    p_dash.add_argument("--publish", action="store_true", help="Publish best model to HF Hub")
    p_dash.add_argument(
        "--remote", default=None, metavar="URL",
        help="Poll a remote bridge server instead of running locally (e.g. http://host:8377)",
    )

    # agent
    p_agent = subparsers.add_parser("agent", help="Run a single agent")
    p_agent.add_argument("--name", required=True, help="Agent name")
    p_agent.add_argument("--message", default=None, help="Custom user message")
    p_agent.add_argument("--temperature", type=float, default=0.1)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_CMD_MAP = {
    "pipeline": cmd_pipeline,
    "sweep": cmd_sweep,
    "experiment": cmd_experiment,
    "quick": cmd_quick,
    "history": cmd_history,
    "bridge": cmd_bridge,
    "client": cmd_client,
    "data": cmd_data,
    "score": cmd_score,
    "dashboard": cmd_dashboard,
    "agent": cmd_agent,
}


def main() -> None:
    """CLI entry point — called by ``synth-city`` console script."""
    print_banner()
    parser = build_parser()
    args = parser.parse_args()
    _CMD_MAP[args.command](args)
