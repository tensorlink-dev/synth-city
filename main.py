"""
synth-city — main entry point for the agentic pipeline.

Usage:
    # Run the full improvement pipeline
    python main.py pipeline

    # Run a quick baseline sweep
    python main.py sweep

    # Run a single experiment
    python main.py experiment --blocks TransformerBlock,LSTMBlock --head SDEHead

    # Query experiment history (Hippius, Trackio, HF Hub)
    python main.py history hippius
    python main.py history trackio --order best --limit 10
    python main.py history hf

    # Start the HTTP bridge server (works standalone or with OpenClaw)
    python main.py bridge

    # Talk to the bridge from the CLI (no OpenClaw needed)
    python main.py client blocks
    python main.py client price BTC
    python main.py client run

    # Run a single agent for debugging
    python main.py agent --name planner
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from logging.handlers import RotatingFileHandler

from config import LOG_DIR, LOG_LEVEL

_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

# Console handler — same as before
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format=_LOG_FORMAT,
    datefmt="%H:%M:%S",
)

# File handler — persists full logs to disk with rotation
_file_handler = RotatingFileHandler(
    LOG_DIR / "synth-city.log",
    maxBytes=20 * 1024 * 1024,  # 20 MB per file
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger("synth-city")


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Run the full agentic improvement pipeline."""
    from pipeline.bootstrap import bootstrap_all
    from pipeline.orchestrator import PipelineOrchestrator

    bootstrap_all()

    loops = getattr(args, "loops", 1)
    max_experiments = getattr(args, "max_experiments", 1)
    for loop_num in range(1, loops + 1):
        if loops > 1:
            logger.info("=== Pipeline loop %d/%d ===", loop_num, loops)
        task: dict = {
            "channel": args.channel,
            "experiment_budget": max_experiments,
        }
        orchestrator = PipelineOrchestrator(
            max_retries=args.retries,
            base_temperature=args.temperature,
            publish=args.publish,
        )
        result = orchestrator.run(task)
        print(json.dumps(result, indent=2, default=str))
        if loops > 1:
            logger.info(
                "Loop %d/%d finished: success=%s",
                loop_num, loops, result.get("success", False),
            )


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run a quick preset sweep via ResearchSession."""
    from pipeline.bootstrap import bootstrap_dirs
    from pipeline.tools.research_tools import _import_research_session

    bootstrap_dirs()
    ResearchSession = _import_research_session()
    session = ResearchSession()
    presets = [p.strip() for p in args.presets.split(",")] if args.presets else None
    result = session.sweep(preset_names=presets, epochs=args.epochs)

    print(json.dumps(result, indent=2, default=str))

    comparison = session.compare()
    print("\n=== RANKING (by CRPS, best first) ===")
    for i, entry in enumerate(comparison.get("ranking", []), 1):
        print(f"  {i}. {entry['name']}  CRPS={entry['crps']:.6f}  params={entry['param_count']}")


def cmd_experiment(args: argparse.Namespace) -> None:
    """Run a single experiment."""
    from pipeline.bootstrap import bootstrap_dirs
    from pipeline.tools.research_tools import _import_research_session

    bootstrap_dirs()
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
        print("Validation FAILED:")
        for err in validation["errors"]:
            print(f"  - {err}")
        sys.exit(1)

    # Run
    run_kwargs: dict = {"epochs": args.epochs}
    if args.early_stopping:
        run_kwargs["early_stopping"] = True
        run_kwargs["patience"] = args.patience
    try:
        result = session.run(experiment, **run_kwargs)
    except TypeError:
        run_kwargs.pop("early_stopping", None)
        run_kwargs.pop("patience", None)
        result = session.run(experiment, **run_kwargs)
    print(json.dumps(result, indent=2, default=str))

    if result.get("status") == "ok":
        metrics = result["metrics"]
        print(f"\nCRPS: {metrics['crps']:.6f}")
        print(f"Sharpness: {metrics['sharpness']:.6f}")
        print(f"Log-likelihood: {metrics['log_likelihood']:.6f}")


def cmd_quick(args: argparse.Namespace) -> None:
    """One-liner convenience experiment."""
    import importlib

    from pipeline.bootstrap import bootstrap_dirs

    bootstrap_dirs()
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
    print(json.dumps(result, indent=2, default=str))


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
                print("Usage: python main.py client price <ASSET>")
                sys.exit(1)
            result = client.get_price(args.extra[0])
        elif action == "history":
            if not args.extra:
                print("Usage: python main.py client history <ASSET> [days]")
                sys.exit(1)
            days = int(args.extra[1]) if len(args.extra) > 1 else 30
            result = client.get_history(args.extra[0], days=days)
        else:
            print(f"Unknown action: {action}")
            print(
                "Available: health blocks heads presets compare"
                " summary clear status run price history"
            )
            sys.exit(1)

        print(json.dumps(result, indent=2, default=str))
    except Exception as exc:
        print(f"Error: {exc}")
        print("Is the bridge running? Start it with: python main.py bridge")
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
        print(json.dumps(data, indent=2, default=str))

        # Print summary table for history queries
        if not args.run_id and "experiments" in data:
            print(f"\n=== HIPPIUS HISTORY ({data.get('total_stored', '?')} total) ===")
            for i, exp in enumerate(data["experiments"], 1):
                crps = exp.get("crps", "N/A")
                crps_str = f"{crps:.6f}" if isinstance(crps, (int, float)) else str(crps)
                blocks = ", ".join(exp.get("blocks", []))
                print(f"  {i}. [{exp.get('run_id', '?')[:15]}] {exp.get('name', '?'):20s}  "
                      f"CRPS={crps_str}  blocks=[{blocks}]  head={exp.get('head', '?')}")

    elif source == "trackio":
        from pipeline.tools.analysis_tools import (
            analyze_experiment_trends,
            fetch_experiment_runs,
        )
        if args.trends:
            result = analyze_experiment_trends(limit=args.limit)
            data = json.loads(result)
            print(json.dumps(data, indent=2, default=str))
            if "timeline" in data:
                print(f"\n=== CRPS TRENDS ({data.get('total_runs', '?')} runs) ===")
                print(f"  Best CRPS:   {data.get('best_crps', 'N/A')}")
                print(f"  Best run:    {data.get('best_run', 'N/A')}")
                print(f"  Latest CRPS: {data.get('latest_crps', 'N/A')}")
                print(f"  Improvement: {data.get('improvement', 'N/A')}")
        else:
            result = fetch_experiment_runs(limit=args.limit, order=args.order)
            data = json.loads(result)
            print(json.dumps(data, indent=2, default=str))
            if "runs" in data:
                print(f"\n=== EXPERIMENT RUNS (order={args.order}) ===")
                for i, run in enumerate(data["runs"], 1):
                    crps = run.get("crps", "N/A")
                    crps_str = f"{crps:.6f}" if isinstance(crps, (int, float)) else str(crps)
                    print(f"  {i}. {run.get('name', '?'):30s}  CRPS={crps_str}")

    elif source == "hf":
        from pipeline.tools.analysis_tools import list_hf_models
        result = list_hf_models(repo_id=args.repo_id or "")
        data = json.loads(result)
        print(json.dumps(data, indent=2, default=str))
        if "files" in data:
            print(f"\n=== HF HUB: {data.get('repo_id', '?')} ===")
            print(f"  Downloads: {data.get('downloads', 'N/A')}")
            print(f"  Likes: {data.get('likes', 'N/A')}")
            print(f"  Files: {len(data.get('files', []))}")
            for f in data.get("files", []):
                print(f"    - {f.get('path', '?')}")

    else:
        print(f"Unknown source: {source}")
        print("Available: hippius, trackio, hf")
        sys.exit(1)


def cmd_results(args: argparse.Namespace) -> None:
    """Share or ingest experiment results via HF Hub Datasets."""
    action = args.action

    if action == "share":
        from pipeline.tools.publish_tools import share_results

        print("Uploading experiment results to HF Hub Dataset…")
        result_json = share_results(repo_id=args.repo_id or "", limit=args.limit)
        data = json.loads(result_json)
        print(json.dumps(data, indent=2, default=str))

        if data.get("status") == "shared":
            print(f"\nShared {data['experiments']} experiments.")
            print(f"URL: {data['url']}")
            print("\nOthers can now consume your results:")
            print('  from datasets import load_dataset')
            print(f'  ds = load_dataset("{data["repo_id"]}")')
        elif data.get("error"):
            print(f"\nError: {data['error']}")

    elif action == "ingest":
        from pipeline.tools.publish_tools import ingest_results

        if not args.repo_id:
            print("Error: --repo-id is required for ingest")
            print("  e.g. synth-city results ingest "
                  "--repo-id tensorlink-dev/synth-city-results")
            sys.exit(1)

        print(f"Downloading results from {args.repo_id}…")
        result_json = ingest_results(repo_id=args.repo_id, limit=args.limit)
        data = json.loads(result_json)
        print(json.dumps(data, indent=2, default=str))

        if data.get("status") == "ingested":
            print(f"\nIngested {data['experiments_saved']} experiments "
                  f"from {data['source']}")
            if data.get("analysis_saved"):
                print("Analysis saved too.")
            print("\nThese will now appear in scan_experiment_history "
                  "and load_hippius_history.")
        elif data.get("error"):
            print(f"\nError: {data['error']}")

    else:
        print(f"Unknown action: {action}")
        print("Available: share, ingest")
        sys.exit(1)


def cmd_display(args: argparse.Namespace) -> None:
    """Display historical experiment results in a terminal dashboard."""
    from cli.history_dashboard import run_display

    run_display(limit=args.limit)


def cmd_score(args: argparse.Namespace) -> None:
    """Scoring emulator — local replica of SN50 validator scoring."""
    action = args.action

    if action == "status":
        from subnet.score_tracker import ScoreTracker
        tracker = ScoreTracker()
        status = tracker.get_status()
        print(json.dumps(status, indent=2, default=str))

    elif action == "results":
        from subnet.score_tracker import ScoreTracker
        tracker = ScoreTracker()
        date = getattr(args, "date", None)
        results = tracker.load_prompt_history(date_str=date, limit=args.limit)
        print(json.dumps(results, indent=2, default=str))

    elif action == "daily":
        from subnet.score_tracker import ScoreTracker
        tracker = ScoreTracker()
        date = getattr(args, "date", None)
        summary = tracker.save_daily_summary(date_str=date)
        print(json.dumps(summary, indent=2, default=str))

    elif action == "leaderboard":
        from subnet.score_tracker import ScoreTracker
        tracker = ScoreTracker()
        lb = tracker.load_leaderboard()
        print(json.dumps(lb, indent=2, default=str))

    elif action == "run":
        from subnet.score_tracker import ScoreTracker, ScoringDaemon
        tracker = ScoreTracker()
        interval = getattr(args, "interval", 5)
        daemon = ScoringDaemon(tracker, interval_minutes=interval)
        logger.info("Starting scoring daemon (interval=%dmin)...", interval)
        logger.info("Press Ctrl+C to stop.")
        daemon.start()
        try:
            while daemon.running:
                import time as _t
                _t.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping scoring daemon...")
            daemon.stop()

    elif action == "backtest":
        from subnet.backtest import ValidatorBacktest, build_baseline_miner

        asset_list: list[str] | None = None
        if args.assets:
            asset_list = [a.strip().upper() for a in args.assets.split(",")]

        logger.info("Building GBM baseline miner for backtest...")
        miner = build_baseline_miner(assets=asset_list)
        if not miner.models:
            print("Error: No models could be registered (no price data available)")
            sys.exit(1)

        bt = ValidatorBacktest(
            miner=miner,
            assets=asset_list,
            interval_minutes=getattr(args, "interval", 60),
            max_prompts=getattr(args, "max_prompts", 50),
            num_paths=getattr(args, "num_paths", 100),
        )
        bt_results = bt.run()

        if "error" in bt_results:
            print(f"Backtest error: {bt_results['error']}")
            sys.exit(1)

        summary = bt_results["summary"]
        wc = summary["weighted_crps"]

        print("\n=== VALIDATOR BACKTEST RESULTS ===")
        print(f"  Prompts: {summary['scored_prompts']}/{summary['total_prompts']} scored")
        print(f"  Interval: {summary['interval_minutes']}min, Paths: {summary['num_paths']}")
        print(f"  Elapsed: {summary['elapsed_seconds']:.1f}s")

        if wc["mean"] is not None:
            print("\n  Weighted CRPS (lower is better):")
            print(f"    Mean:   {wc['mean']:.4f}")
            print(f"    Median: {wc['median']:.4f}")
            print(f"    Best:   {wc['best']:.4f}")
            print(f"    Worst:  {wc['worst']:.4f}")
            print(f"    Std:    {wc['std']:.4f}")

        if summary["per_asset"]:
            print("\n  Per-asset CRPS breakdown:")
            print(f"    {'Asset':<10} {'Weight':<8} {'Mean':<12} {'Median':<12} {'Best':<12}")
            print(f"    {'─' * 54}")
            for asset, stats in sorted(summary["per_asset"].items()):
                print(
                    f"    {asset:<10} {stats['weight']:<8.2f} "
                    f"{stats['mean_crps']:<12.4f} {stats['median_crps']:<12.4f} "
                    f"{stats['best_crps']:<12.4f}"
                )

        # Also dump full JSON for programmatic use
        print("\n--- Full JSON ---")
        print(json.dumps(summary, indent=2, default=str))

    else:
        print(f"Unknown action: {action}")
        print("Available: run, status, results, daily, leaderboard, backtest")
        sys.exit(1)


def cmd_agent(args: argparse.Namespace) -> None:
    """Run a single agent for debugging/testing."""
    from pipeline.bootstrap import bootstrap_dirs

    bootstrap_dirs()

    from pipeline.agents.agent_designer import AgentDesignerAgent
    from pipeline.agents.author import ComponentAuthorAgent
    from pipeline.agents.code_checker import CodeCheckerAgent
    from pipeline.agents.debugger import DebuggerAgent
    from pipeline.agents.pipeline_architect import PipelineArchitectAgent
    from pipeline.agents.planner import PlannerAgent
    from pipeline.agents.publisher import PublisherAgent
    from pipeline.agents.trainer import TrainerAgent
    from pipeline.orchestrator import resolve_agent

    agents = {
        "planner": PlannerAgent,
        "codechecker": CodeCheckerAgent,
        "debugger": DebuggerAgent,
        "trainer": TrainerAgent,
        "publisher": PublisherAgent,
        "author": ComponentAuthorAgent,
        "agent_designer": AgentDesignerAgent,
        "pipeline_architect": PipelineArchitectAgent,
    }

    agent_cls = agents.get(args.name.lower())
    # Fall back to dynamic resolution for authored agents
    if not agent_cls:
        agent_cls = resolve_agent(args.name.lower())
    if not agent_cls:
        logger.error("Unknown agent: %s (available: %s)", args.name, list(agents.keys()))
        sys.exit(1)

    task: dict = {
        "channel": "default",
        "user_message": args.message or "Begin the task.",
    }

    agent = agent_cls(temperature=args.temperature)
    result = agent.run(task)
    print(f"\nAgent: {args.name}")
    print(f"Success: {result.success}")
    print(f"Turns: {result.turns_used}")
    if result.structured:
        print(f"Structured output:\n{json.dumps(result.structured, indent=2, default=str)}")
    if result.raw_text:
        print(f"Raw text:\n{result.raw_text[:2000]}")


def main() -> None:
    from cli.display import print_banner

    print_banner()

    parser = argparse.ArgumentParser(
        description="synth-city — agentic pipeline for Bittensor SN50 competition"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Run the full improvement pipeline")
    p_pipe.add_argument("--channel", default="default", help="Prompt channel")
    p_pipe.add_argument("--retries", type=int, default=5, help="Max retries per stage")
    p_pipe.add_argument("--temperature", type=float, default=0.1, help="Base LLM temperature")
    p_pipe.add_argument("--publish", action="store_true", help="Publish best model to HF Hub")
    p_pipe.add_argument(
        "--loops", type=int, default=1,
        help="Number of pipeline runs (each gets a fresh orchestrator and run ID)",
    )
    p_pipe.add_argument(
        "--max-experiments", type=int, default=1, dest="max_experiments",
        help="Max experiments the trainer may run per pipeline loop (default: 1)",
    )

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Run a preset sweep")
    p_sweep.add_argument(
        "--presets", default=None,
        help="Comma-separated preset names (all if omitted)",
    )
    p_sweep.add_argument("--epochs", type=int, default=10, help="Epochs per preset")
    p_sweep.add_argument(
        "--early-stopping", action="store_true", default=True,
        help="Enable early stopping (default: true)",
    )
    p_sweep.add_argument(
        "--no-early-stopping", dest="early_stopping",
        action="store_false", help="Disable early stopping",
    )
    p_sweep.add_argument("--patience", type=int, default=3, help="Early stopping patience")

    # experiment
    p_exp = subparsers.add_parser("experiment", help="Run a single experiment")
    p_exp.add_argument("--blocks", required=True, help="Comma-separated block names")
    p_exp.add_argument("--head", default="GBMHead", help="Head name")
    p_exp.add_argument("--d-model", type=int, default=32, help="Hidden dimension")
    p_exp.add_argument("--horizon", type=int, default=12, help="Prediction steps")
    p_exp.add_argument("--n-paths", type=int, default=100, help="Monte Carlo paths")
    p_exp.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    p_exp.add_argument("--epochs", type=int, default=10, help="Training epochs")
    p_exp.add_argument(
        "--early-stopping", action="store_true", default=True,
        help="Enable early stopping (default: true)",
    )
    p_exp.add_argument(
        "--no-early-stopping", dest="early_stopping",
        action="store_false", help="Disable early stopping",
    )
    p_exp.add_argument("--patience", type=int, default=3, help="Early stopping patience")

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
        choices=["health", "blocks", "heads", "presets", "compare", "summary",
                 "clear", "status", "run", "price", "history"],
        help="Action to perform",
    )
    p_client.add_argument("extra", nargs="*", help="Extra args (e.g. asset name for price/history)")
    p_client.add_argument("--host", default=BRIDGE_HOST, help="Bridge host")
    p_client.add_argument("--port", type=int, default=BRIDGE_PORT, help="Bridge port")
    p_client.add_argument("--publish", action="store_true", help="Publish when using 'run' action")

    # history
    p_hist = subparsers.add_parser(
        "history",
        help="Query experiment history (Hippius, Trackio, HF Hub)",
    )
    p_hist.add_argument("source", choices=["hippius", "trackio", "hf"], help="Data source")
    p_hist.add_argument("--limit", type=int, default=20, help="Max results to return")
    p_hist.add_argument("--order", default="best", choices=["best", "recent", "worst"],
                        help="Sort order for experiment runs")
    p_hist.add_argument("--trends", action="store_true",
                        help="Show CRPS trends (trackio only)")
    p_hist.add_argument(
        "--run-id", default=None,
        help="Load a specific Hippius run ID ('latest' for most recent)",
    )
    p_hist.add_argument("--repo-id", default=None, help="HF Hub repo ID override")

    # results (public sharing / ingestion)
    p_results = subparsers.add_parser(
        "results",
        help="Share or ingest experiment results via HF Hub Datasets",
    )
    p_results.add_argument(
        "action", choices=["share", "ingest"],
        help="'share' uploads your results, 'ingest' pulls someone else's",
    )
    p_results.add_argument(
        "--repo-id", default=None,
        help="HF dataset repo ID. Required for ingest, optional for share",
    )
    p_results.add_argument(
        "--limit", type=int, default=200,
        help="Max experiments to include/ingest (default 200)",
    )

    # display
    p_display = subparsers.add_parser(
        "display", help="Display historical experiment results dashboard"
    )
    p_display.add_argument(
        "--limit", type=int, default=50,
        help="Max experiments to fetch from storage (default 50)",
    )

    # score
    p_score = subparsers.add_parser(
        "score", help="Scoring emulator — local replica of SN50 validator scoring"
    )
    p_score.add_argument(
        "action",
        choices=["run", "status", "results", "daily", "leaderboard", "backtest"],
        help="Action: run (start daemon), status, results, daily, leaderboard, backtest",
    )
    p_score.add_argument("--interval", type=int, default=5, help="Prompt interval in minutes")
    p_score.add_argument("--date", default=None, help="Date filter (YYYY-MM-DD)")
    p_score.add_argument("--limit", type=int, default=20, help="Max results to return")
    p_score.add_argument("--assets", default=None, help="Backtest: comma-separated asset list")
    p_score.add_argument(
        "--max-prompts", type=int, default=50, dest="max_prompts",
        help="Backtest: max prompts to evaluate (default: 50)",
    )
    p_score.add_argument(
        "--num-paths", type=int, default=100, dest="num_paths",
        help="Backtest: Monte Carlo paths per prediction (default: 100)",
    )

    # agent
    p_agent = subparsers.add_parser("agent", help="Run a single agent")
    p_agent.add_argument("--name", required=True, help="Agent name")
    p_agent.add_argument("--message", default=None, help="Custom user message")
    p_agent.add_argument("--temperature", type=float, default=0.1)

    args = parser.parse_args()
    cmd_map = {
        "pipeline": cmd_pipeline,
        "sweep": cmd_sweep,
        "experiment": cmd_experiment,
        "quick": cmd_quick,
        "history": cmd_history,
        "results": cmd_results,
        "display": cmd_display,
        "bridge": cmd_bridge,
        "client": cmd_client,
        "score": cmd_score,
        "agent": cmd_agent,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
