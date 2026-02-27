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
    "publisher": os.getenv("PUBLISHER_MODEL", DEFAULT_MODEL),
    "author": os.getenv("AUTHOR_MODEL", "Qwen/Qwen3-235B-A22B"),
}


def model_for(agent_name: str) -> str:
    """Return the model ID to use for *agent_name*, falling back to DEFAULT_MODEL."""
    return AGENT_MODELS.get(agent_name.lower(), DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# Basilica GPU cloud (basilica-sdk)
# ---------------------------------------------------------------------------
BASILICA_API_TOKEN: str = os.getenv("BASILICA_API_TOKEN", "") or os.getenv("BASILICA_API_KEY", "")
BASILICA_API_URL: str = os.getenv("BASILICA_API_URL", "") or os.getenv("BASILICA_ENDPOINT", "https://api.basilica.ai")
# Budget cap — only rent GPUs at or below this hourly rate (USD)
BASILICA_MAX_HOURLY_RATE: float = float(os.getenv("BASILICA_MAX_HOURLY_RATE", "0.44"))
# Allowed GPU types (case-insensitive substring match against offering gpu_type)
BASILICA_ALLOWED_GPU_TYPES: list[str] = [
    s.strip()
    for s in os.getenv("BASILICA_ALLOWED_GPU_TYPES", "TESLA V100,RTX-A4000,RTX-A6000").split(",")
    if s.strip()
]
# Deployment-based training — Docker image with open-synth-miner pre-installed
BASILICA_DEPLOY_IMAGE: str = os.getenv(
    "BASILICA_DEPLOY_IMAGE", "ghcr.io/tensorlink-ai/synth-city-gpu:latest"
)
BASILICA_DEPLOY_GPU_MODELS: list[str] = [
    s.strip()
    for s in os.getenv("BASILICA_DEPLOY_GPU_MODELS", "").split(",")
    if s.strip()
]
BASILICA_DEPLOY_MIN_GPU_MEMORY_GB: int = int(
    os.getenv("BASILICA_DEPLOY_MIN_GPU_MEMORY_GB", "12")
)
# Pod resource requests (lower = easier to schedule, but may OOM during training)
BASILICA_DEPLOY_CPU: str = os.getenv("BASILICA_DEPLOY_CPU", "2000m")
BASILICA_DEPLOY_MEMORY: str = os.getenv("BASILICA_DEPLOY_MEMORY", "8Gi")
BASILICA_DEPLOY_STORAGE: str = os.getenv("BASILICA_DEPLOY_STORAGE", "10Gi")

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
# HF training data (parquet datasets for MarketDataLoader)
# ---------------------------------------------------------------------------
HF_TRAINING_DATA_REPO: str = os.getenv(
    "HF_TRAINING_DATA_REPO", "tensorlink-dev/open-synth-training-data"
)

# Maps SN50 asset names → HF dataset asset names
SN50_TO_HF_ASSET: dict[str, str] = {
    "BTC": "BTC_USD",
    "ETH": "ETH_USD",
    "SOL": "SOL_USD",
    "SPYX": "SPY",
    "NVDAX": "NVDA",
    "TSLAX": "TSLA",
    "AAPLX": "AAPL",
    "GOOGLX": "GOOGL",
}

# Per-timeframe configuration
TIMEFRAME_CONFIGS: dict[str, dict[str, int | str]] = {
    "5m": {
        "pred_len": 288,       # 24 h ÷ 5 min = 288 steps
        "input_len": 288,
        "file_suffix": "5m.parquet",
    },
    "1m": {
        "pred_len": 60,        # 1 h ÷ 1 min = 60 steps
        "input_len": 60,
        "file_suffix": "1m.parquet",
    },
}

# Data loader defaults
DATA_BATCH_SIZE: int = int(os.getenv("DATA_BATCH_SIZE", "64"))
DATA_STRIDE: int = int(os.getenv("DATA_STRIDE", "12"))
DATA_GAP_HANDLING: str = os.getenv("DATA_GAP_HANDLING", "ffill")
DATA_FEATURE_ENGINEER: str = os.getenv("DATA_FEATURE_ENGINEER", "zscore")

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
# open-synth-miner research defaults
# ---------------------------------------------------------------------------
# Research mode uses fewer paths for speed; production uses 1000
RESEARCH_N_PATHS: int = int(os.getenv("RESEARCH_N_PATHS", "100"))
RESEARCH_D_MODEL: int = int(os.getenv("RESEARCH_D_MODEL", "32"))
RESEARCH_HORIZON: int = int(os.getenv("RESEARCH_HORIZON", "12"))
RESEARCH_SEQ_LEN: int = int(os.getenv("RESEARCH_SEQ_LEN", "32"))
RESEARCH_FEATURE_DIM: int = int(os.getenv("RESEARCH_FEATURE_DIM", "4"))
RESEARCH_BATCH_SIZE: int = int(os.getenv("RESEARCH_BATCH_SIZE", "4"))
RESEARCH_LR: float = float(os.getenv("RESEARCH_LR", "0.001"))
RESEARCH_EPOCHS: int = int(os.getenv("RESEARCH_EPOCHS", "1"))

# ---------------------------------------------------------------------------
# Publishing (HF Hub + W&B + Trackio)
# ---------------------------------------------------------------------------
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
HF_REPO_ID: str = os.getenv("HF_REPO_ID", "")
WANDB_PROJECT: str = os.getenv("WANDB_PROJECT", "synth-city")
TRACKIO_PROJECT: str = os.getenv("TRACKIO_PROJECT", "synth-city")

# ---------------------------------------------------------------------------
# Hippius decentralised storage (S3-compatible)
# ---------------------------------------------------------------------------
HIPPIUS_ENDPOINT: str = os.getenv("HIPPIUS_ENDPOINT", "")
HIPPIUS_ACCESS_KEY: str = os.getenv("HIPPIUS_ACCESS_KEY", "")
HIPPIUS_SECRET_KEY: str = os.getenv("HIPPIUS_SECRET_KEY", "")
HIPPIUS_BUCKET: str = os.getenv("HIPPIUS_BUCKET", "synth-city")

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
MAX_AGENT_TURNS: int = int(os.getenv("MAX_AGENT_TURNS", "50"))
WORKSPACE_DIR: Path = Path(os.getenv("WORKSPACE_DIR", "./workspace"))
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR: Path = Path(os.getenv("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# OpenClaw bridge
# ---------------------------------------------------------------------------
BRIDGE_HOST: str = os.getenv("BRIDGE_HOST", "127.0.0.1")
BRIDGE_PORT: int = int(os.getenv("BRIDGE_PORT", "8377"))
BRIDGE_API_KEY: str = os.getenv("BRIDGE_API_KEY", "")

# ---------------------------------------------------------------------------
# Multi-bot concurrency
# ---------------------------------------------------------------------------
BOT_SESSION_TTL_SECONDS: int = int(os.getenv("BOT_SESSION_TTL_SECONDS", "3600"))
MAX_CONCURRENT_PIPELINES: int = int(os.getenv("MAX_CONCURRENT_PIPELINES", "10"))
