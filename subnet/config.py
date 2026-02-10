"""SN50-specific configuration and constants."""

from __future__ import annotations

from config import (
    SN50_ASSETS,
    SN50_HFT_HORIZON_MINUTES,
    SN50_HORIZON_MINUTES,
    SN50_NUM_PATHS,
    SN50_STEP_MINUTES,
)

# Number of time steps in 24h horizon (including t0)
NUM_STEPS_24H: int = (SN50_HORIZON_MINUTES // SN50_STEP_MINUTES) + 1  # 289

# Number of time steps in HFT 1h horizon (including t0)
NUM_STEPS_HFT: int = (SN50_HFT_HORIZON_MINUTES // SN50_STEP_MINUTES) + 1  # 13

# Time increment array for CRPS evaluation (in minutes)
CRPS_EVAL_INCREMENTS: list[int] = [5, 10, 15, 30, 60, 180, 360, 720, 1440]

# Softmax beta for emission allocation
EMISSION_SOFTMAX_BETA: float = -0.1

# Leaderboard rolling window (days)
LEADERBOARD_WINDOW_DAYS: int = 10

# Score capping percentile
SCORE_CAP_PERCENTILE: float = 0.90
