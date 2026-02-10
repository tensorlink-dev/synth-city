"""
synth-city — main entry point for the agentic pipeline.

Usage:
    # Run the full improvement pipeline
    python main.py pipeline --assets BTC,ETH,SOL

    # Generate predictions with current models
    python main.py predict --assets BTC,ETH --horizon 24h

    # Evaluate a model locally
    python main.py evaluate --model workspace/model.py --asset BTC

    # Run a single agent
    python main.py agent --name planner --assets BTC
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

    assets = [a.strip().upper() for a in args.assets.split(",")]
    task = {
        "target_assets": assets,
        "channel": args.channel,
    }
    if args.model:
        task["current_model_path"] = args.model

    orchestrator = PipelineOrchestrator(
        max_retries=args.retries,
        base_temperature=args.temperature,
    )
    result = orchestrator.run(task)
    print(json.dumps(result, indent=2, default=str))


def cmd_predict(args: argparse.Namespace) -> None:
    """Generate predictions using fitted models."""
    from data.market import get_close_prices, get_latest_prices
    from models.garch import GARCHForecaster
    from subnet.miner import SynthMiner

    assets = [a.strip().upper() for a in args.assets.split(",")]
    miner = SynthMiner()

    # Fit models and generate predictions
    prices = get_latest_prices(assets)
    for asset in assets:
        logger.info("Fitting GARCH model for %s...", asset)
        model = GARCHForecaster(variant="GJR-GARCH", dist="t")
        try:
            historical = get_close_prices(asset, days=30)
            model.fit(historical)
            miner.register_model(asset, model)
            if asset in prices:
                miner.set_price(asset, prices[asset])
        except Exception as exc:
            logger.error("Failed to fit model for %s: %s", asset, exc)

    predictions = miner.generate_all_predictions(horizon=args.horizon)
    submission = miner.format_submission(predictions)

    # Save
    out_path = args.output or "predictions.json"
    with open(out_path, "w") as f:
        json.dump(submission, f)
    logger.info("Saved predictions to %s (%d assets)", out_path, len(predictions))


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a model locally against recent price data."""
    import numpy as np

    from data.market import get_close_prices
    from subnet.validator import evaluate_prediction

    # For local evaluation, we use historical data as "realized" prices
    historical = get_close_prices(args.asset, days=7)
    if len(historical) < 300:
        logger.error("Not enough data for evaluation (need >= 300 points, got %d)", len(historical))
        sys.exit(1)

    # Split into calibration and evaluation
    cal_data = historical[:-289]
    realized = historical[-289:]

    # Fit a model on calibration data
    from models.garch import GARCHForecaster

    model = GARCHForecaster(variant=args.variant, dist="t")
    model.fit(cal_data)

    # Generate paths
    paths = model.generate_paths(
        asset=args.asset,
        num_paths=1000,
        num_steps=289,
        s0=float(realized[0]),
    )

    # Score
    scores = evaluate_prediction(paths, realized)
    print(f"\nEvaluation results for {args.asset} ({args.variant}):")
    print("-" * 50)
    for key, val in sorted(scores.items()):
        print(f"  {key:20s}: {val:12.4f}")


def cmd_agent(args: argparse.Namespace) -> None:
    """Run a single agent for debugging/testing."""
    from pipeline.agents.code_checker import CodeCheckerAgent
    from pipeline.agents.debugger import DebuggerAgent
    from pipeline.agents.planner import PlannerAgent
    from pipeline.agents.trainer import TrainerAgent

    agents = {
        "planner": PlannerAgent,
        "codechecker": CodeCheckerAgent,
        "debugger": DebuggerAgent,
        "trainer": TrainerAgent,
    }

    agent_cls = agents.get(args.name.lower())
    if not agent_cls:
        logger.error("Unknown agent: %s (available: %s)", args.name, list(agents.keys()))
        sys.exit(1)

    assets = [a.strip().upper() for a in args.assets.split(",")]
    task = {
        "target_assets": assets,
        "channel": "default",
        "user_message": args.message or "Begin the task.",
    }
    if args.model:
        task["model_path"] = args.model
        task["current_model_path"] = args.model

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
    p_pipe.add_argument("--assets", default="BTC,ETH,SOL", help="Comma-separated asset list")
    p_pipe.add_argument("--model", default=None, help="Path to current model file")
    p_pipe.add_argument("--channel", default="default", help="Prompt channel")
    p_pipe.add_argument("--retries", type=int, default=5, help="Max retries per stage")
    p_pipe.add_argument("--temperature", type=float, default=0.1, help="Base LLM temperature")

    # predict
    p_pred = subparsers.add_parser("predict", help="Generate predictions")
    p_pred.add_argument("--assets", default="BTC,ETH,SOL", help="Comma-separated asset list")
    p_pred.add_argument("--horizon", default="24h", choices=["24h", "1h"])
    p_pred.add_argument("--output", default=None, help="Output file path")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a model locally")
    p_eval.add_argument("--asset", required=True, help="Asset to evaluate")
    p_eval.add_argument("--variant", default="GJR-GARCH", help="GARCH variant")

    # agent
    p_agent = subparsers.add_parser("agent", help="Run a single agent")
    p_agent.add_argument("--name", required=True, help="Agent name")
    p_agent.add_argument("--assets", default="BTC", help="Comma-separated asset list")
    p_agent.add_argument("--model", default=None, help="Model file path")
    p_agent.add_argument("--message", default=None, help="Custom user message")
    p_agent.add_argument("--temperature", type=float, default=0.1)

    args = parser.parse_args()
    cmd_map = {
        "pipeline": cmd_pipeline,
        "predict": cmd_predict,
        "evaluate": cmd_evaluate,
        "agent": cmd_agent,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
