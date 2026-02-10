"""Global configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Chutes AI (LLM inference)
# ---------------------------------------------------------------------------
CHUTES_API_KEY: str = os.getenv("CHUTES_API_KEY", "")
CHUTES_BASE_URL: str = os.getenv("CHUTES_BASE_URL", "https://llm.chutes.ai/v1")

# ---------------------------------------------------------------------------
# Per-agent model selection
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")

AGENT_MODELS: dict[str, str] = {
    "planner": os.getenv("PLANNER_MODEL", "Qwen/Qwen3-235B-A22B"),
    "codechecker": os.getenv("CODECHECKER_MODEL", DEFAULT_MODEL),
    "debugger": os.getenv("DEBUGGER_MODEL", DEFAULT_MODEL),
    "trainer": os.getenv("TRAINER_MODEL", "Qwen/Qwen3-235B-A22B"),
}


def model_for(agent_name: str) -> str:
    """Return the model ID to use for *agent_name*, falling back to DEFAULT_MODEL."""
    return AGENT_MODELS.get(agent_name.lower(), DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# Basilica compute (training)
# ---------------------------------------------------------------------------
BASILICA_API_KEY: str = os.getenv("BASILICA_API_KEY", "")
BASILICA_ENDPOINT: str = os.getenv("BASILICA_ENDPOINT", "https://api.basilica.tplr.ai")

# ---------------------------------------------------------------------------
# Bittensor
# ---------------------------------------------------------------------------
BT_WALLET_NAME: str = os.getenv("BT_WALLET_NAME", "default")
BT_HOTKEY_NAME: str = os.getenv("BT_HOTKEY_NAME", "default")
BT_NETWORK: str = os.getenv("BT_NETWORK", "finney")
BT_NETUID: int = int(os.getenv("BT_NETUID", "50"))

# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
PYTH_PRICE_FEED_URL: str = os.getenv("PYTH_PRICE_FEED_URL", "https://hermes.pyth.network")

# ---------------------------------------------------------------------------
# SN50 asset configuration (asset -> scoring weight)
# ---------------------------------------------------------------------------
SN50_ASSETS: dict[str, float] = {
    "BTC": 1.00,
    "ETH": 0.67,
    "SOL": 0.59,
    "XAU": 2.26,
    "SPYX": 2.99,
    "NVDAX": 1.39,
    "TSLAX": 1.42,
    "AAPLX": 1.86,
    "GOOGLX": 1.43,
}

# Number of simulated paths required per asset
SN50_NUM_PATHS: int = 1000
# Forecast horizon in minutes
SN50_HORIZON_MINUTES: int = 24 * 60  # 24 hours
# Step size in minutes
SN50_STEP_MINUTES: int = 5
# HFT horizon in minutes
SN50_HFT_HORIZON_MINUTES: int = 60

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
MAX_AGENT_TURNS: int = int(os.getenv("MAX_AGENT_TURNS", "50"))
WORKSPACE_DIR: Path = Path(os.getenv("WORKSPACE_DIR", "./workspace"))
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
