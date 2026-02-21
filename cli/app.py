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
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from cli.display import (
    agent_result_panel,
    console,
    hf_panel,
    hippius_table,
    metrics_panel,
    print_banner,
    print_error,
    print_json,
    print_validation_errors,
    ranking_table,
    section_header,
    wandb_runs_table,
    wandb_trends_panel,
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
    from src.research.agent_api import ResearchSession

    from pipeline.bootstrap import bootstrap_dirs

    bootstrap_dirs()
    section_header("Sweep")
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
    from src.research.agent_api import ResearchSession

    from pipeline.bootstrap import bootstrap_dirs

    bootstrap_dirs()
    section_header("Experiment")
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
    from src.research.agent_api import quick_experiment

    section_header("Quick Experiment")
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
    """Query experiment history from Hippius, W&B, or HF Hub."""
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

    elif source == "wandb":
        from pipeline.tools.analysis_tools import analyze_wandb_trends, fetch_wandb_runs

        if args.trends:
            result = analyze_wandb_trends(limit=args.limit)
            data = json.loads(result)
            print_json(data)
            if "timeline" in data:
                console.print()
                wandb_trends_panel(data)
        else:
            result = fetch_wandb_runs(limit=args.limit, order=args.order)
            data = json.loads(result)
            print_json(data)
            if "runs" in data:
                console.print()
                wandb_runs_table(data["runs"], order=args.order)

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
        console.print("[muted]Available: hippius, wandb, hf[/muted]")
        sys.exit(1)


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
        "history", help="Query experiment history (Hippius, W&B, HF Hub)"
    )
    p_hist.add_argument("source", choices=["hippius", "wandb", "hf"], help="Data source")
    p_hist.add_argument("--limit", type=int, default=20, help="Max results to return")
    p_hist.add_argument(
        "--order", default="best", choices=["best", "recent", "worst"],
        help="Sort order for W&B runs",
    )
    p_hist.add_argument("--trends", action="store_true", help="Show CRPS trends (W&B only)")
    p_hist.add_argument(
        "--run-id", default=None,
        help="Load a specific Hippius run ID ('latest' for most recent)",
    )
    p_hist.add_argument("--repo-id", default=None, help="HF Hub repo ID override")

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
    "agent": cmd_agent,
}


def main() -> None:
    """CLI entry point — called by ``synth-city`` console script."""
    print_banner()
    parser = build_parser()
    args = parser.parse_args()
    _CMD_MAP[args.command](args)
