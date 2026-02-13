"""
synth-city — main entry point for the agentic pipeline.

Usage:
    # Run the full improvement pipeline
    python main.py pipeline

    # Run a quick baseline sweep
    python main.py sweep

    # Run a single experiment
    python main.py experiment --blocks TransformerBlock,LSTMBlock --head SDEHead

    # Start the OpenClaw bridge server
    python main.py bridge

    # Run a single agent for debugging
    python main.py agent --name planner
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("synth-city")


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Run the full agentic improvement pipeline."""
    from pipeline.orchestrator import PipelineOrchestrator

    task: dict = {"channel": args.channel}
    orchestrator = PipelineOrchestrator(
        max_retries=args.retries,
        base_temperature=args.temperature,
        publish=args.publish,
    )
    result = orchestrator.run(task)
    print(json.dumps(result, indent=2, default=str))


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run a quick preset sweep via ResearchSession."""
    from src.research.agent_api import ResearchSession

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
    from src.research.agent_api import ResearchSession

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
    result = session.run(experiment, epochs=args.epochs)
    print(json.dumps(result, indent=2, default=str))

    if result.get("status") == "ok":
        metrics = result["metrics"]
        print(f"\nCRPS: {metrics['crps']:.6f}")
        print(f"Sharpness: {metrics['sharpness']:.6f}")
        print(f"Log-likelihood: {metrics['log_likelihood']:.6f}")


def cmd_quick(args: argparse.Namespace) -> None:
    """One-liner convenience experiment."""
    from src.research.agent_api import quick_experiment

    blocks = [b.strip() for b in args.blocks.split(",")] if args.blocks else None
    result = quick_experiment(
        blocks=blocks,
        head=args.head,
        d_model=args.d_model,
        horizon=args.horizon,
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_bridge(args: argparse.Namespace) -> None:
    """Start the OpenClaw bridge HTTP server."""
    from integrations.openclaw.bridge import run_bridge

    run_bridge(host=args.host, port=args.port)


def cmd_agent(args: argparse.Namespace) -> None:
    """Run a single agent for debugging/testing."""
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
    }

    agent_cls = agents.get(args.name.lower())
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

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Run a preset sweep")
    p_sweep.add_argument("--presets", default=None, help="Comma-separated preset names (all if omitted)")
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

    # bridge (OpenClaw integration)
    p_bridge = subparsers.add_parser("bridge", help="Start the OpenClaw bridge HTTP server")
    p_bridge.add_argument("--host", default="127.0.0.1", help="Bind address")
    p_bridge.add_argument("--port", type=int, default=8377, help="Listen port")

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
        "bridge": cmd_bridge,
        "agent": cmd_agent,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
