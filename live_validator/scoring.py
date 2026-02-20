"""
CRPS scoring — matches the synth-subnet validator's interval-based CRPS logic.

Implements:
    - Per-interval CRPS calculation (5min, 30min, 3h, 24h_abs for low-freq)
    - Raw CRPS scores (no softmax normalisation — we just want the numbers)

References:
    synth-subnet/synth/validator/crps_calculation.py
    synth-subnet/synth/validator/prompt_config.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from properscoring import crps_ensemble

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring interval configs — mirrors synth-subnet prompt_config.py exactly
# ---------------------------------------------------------------------------
LOW_FREQ_SCORING_INTERVALS: dict[str, int] = {
    "5min": 300,
    "30min": 1800,
    "3hour": 10800,
    "24hour_abs": 86400,
}

HIGH_FREQ_SCORING_INTERVALS: dict[str, int] = {
    "1min": 60,
    "2min": 120,
    "5min": 300,
    "15min": 900,
    "30min": 1800,
    "60min_abs": 3600,
    "0_5min_gaps": 300,
    "0_10min_gaps": 600,
    "0_15min_gaps": 900,
    "0_20min_gaps": 1200,
    "0_25min_gaps": 1500,
    "0_30min_gaps": 1800,
    "0_35min_gaps": 2100,
    "0_40min_gaps": 2400,
    "0_45min_gaps": 2700,
    "0_50min_gaps": 3000,
    "0_55min_gaps": 3300,
    "0_60min_gaps": 3600,
}

# Per-asset CRPS weighting coefficients — from synth-subnet moving_average.py
ASSET_COEFFICIENTS: dict[str, float] = {
    "BTC": 1.0,
    "ETH": 0.6715516528608204,
    "XAU": 2.262003561659039,
    "SOL": 0.5883682889710361,
    "SPYX": 2.9914378891824693,
    "NVDAX": 1.3885444209082594,
    "TSLAX": 1.420016421725336,
    "AAPLX": 1.864976360560554,
    "GOOGLX": 1.4310534797250312,
}


@dataclass
class ScoringConfig:
    """Configuration for a scoring mode (low or high frequency)."""
    label: str
    time_length: int        # Total prediction horizon in seconds
    time_increment: int     # Seconds per step
    scoring_intervals: dict[str, int]
    num_simulations: int = 1000
    window_days: int = 10


LOW_FREQUENCY = ScoringConfig(
    label="low",
    time_length=86400,      # 24 hours
    time_increment=300,     # 5 minutes
    scoring_intervals=LOW_FREQ_SCORING_INTERVALS,
    window_days=10,
)

HIGH_FREQUENCY = ScoringConfig(
    label="high",
    time_length=3600,       # 1 hour
    time_increment=60,      # 1 minute
    scoring_intervals=HIGH_FREQ_SCORING_INTERVALS,
    window_days=3,
)


# ---------------------------------------------------------------------------
# Core CRPS calculation — mirrors synth-subnet crps_calculation.py
# ---------------------------------------------------------------------------

def get_interval_steps(scoring_interval: int, time_increment: int) -> int:
    """Calculate number of steps in a scoring interval."""
    return int(scoring_interval / time_increment)


def label_observed_blocks(arr: np.ndarray) -> np.ndarray:
    """Group blocks of consecutive observed (non-NaN) data.

    Example: [1.0, 2.0, nan, 4.0] -> [0, 0, -1, 1]
    """
    not_nan = ~np.isnan(arr)
    block_start = not_nan & np.concatenate(([True], ~not_nan[:-1]))
    group_numbers = np.cumsum(block_start) - 1
    return np.where(not_nan, group_numbers, -1)


def calculate_price_changes_over_intervals(
    price_paths: np.ndarray,
    interval_steps: int,
    absolute_price: bool = False,
    is_gap: bool = False,
) -> np.ndarray:
    """Calculate price changes over specified intervals.

    Mirrors synth-subnet crps_calculation.calculate_price_changes_over_intervals.
    """
    interval_prices = price_paths[:, ::interval_steps]
    if is_gap:
        interval_prices = interval_prices[:1]

    if absolute_price:
        return interval_prices[:, 1:]

    return (np.diff(interval_prices, axis=1) / interval_prices[:, :-1]) * 10_000


def calculate_crps_for_miner(
    simulation_runs: np.ndarray,
    real_price_path: np.ndarray,
    time_increment: int,
    scoring_intervals: dict[str, int],
) -> tuple[float, list[dict]]:
    """Calculate total CRPS score for a model's simulations over all intervals.

    This is a direct port of synth-subnet's calculate_crps_for_miner.

    Parameters
    ----------
    simulation_runs : np.ndarray
        Shape (num_sims, num_steps) — simulated price paths.
    real_price_path : np.ndarray
        Shape (num_steps,) — the actual price path.
    time_increment : int
        Seconds between time steps.
    scoring_intervals : dict
        Interval name -> duration in seconds.

    Returns
    -------
    (total_crps, detailed_data)
    """
    detailed_crps_data: list[dict] = []
    sum_all_scores = 0.0

    for interval_name, interval_seconds in scoring_intervals.items():
        interval_steps = get_interval_steps(interval_seconds, time_increment)
        absolute_price = interval_name.endswith("_abs")
        is_gap = (
            interval_name.endswith("_gap")
            or interval_name.endswith("_gaps")
        )

        if absolute_price:
            while (
                real_price_path[::interval_steps].shape[0] == 1
                and interval_steps > 1
            ):
                interval_steps -= 1

        # Check for zero prices
        if np.any(simulation_runs == 0):
            return -1.0, [{"error": "Zero price in simulation runs"}]

        simulated_changes = calculate_price_changes_over_intervals(
            simulation_runs, interval_steps, absolute_price, is_gap,
        )
        real_changes = calculate_price_changes_over_intervals(
            real_price_path.reshape(1, -1),
            interval_steps, absolute_price, is_gap,
        )
        data_blocks = label_observed_blocks(real_changes[0])

        if len(data_blocks) == 0:
            continue

        total_increment = 0
        crps_values = 0.0
        for block in np.unique(data_blocks):
            if block == -1:
                continue

            mask = data_blocks == block
            simulated_block = simulated_changes[:, mask]
            real_block = real_changes[0, mask]
            num_intervals = simulated_block.shape[1]

            crps_values_block = np.array([
                crps_ensemble(real_block[t], simulated_block[:, t])
                for t in range(num_intervals)
            ])

            if absolute_price:
                crps_values_block = (
                    crps_values_block / real_price_path[-1] * 10_000
                )

            crps_values += crps_values_block.sum()

            for t in range(num_intervals):
                detailed_crps_data.append({
                    "Interval": interval_name,
                    "Increment": total_increment + 1,
                    "CRPS": float(crps_values_block[t]),
                })
                total_increment += 1

        total_crps_interval = crps_values
        sum_all_scores += float(total_crps_interval)

        detailed_crps_data.append({
            "Interval": interval_name,
            "Increment": "Total",
            "CRPS": float(total_crps_interval),
        })

    detailed_crps_data.append({
        "Interval": "Overall",
        "Increment": "Total",
        "CRPS": sum_all_scores,
    })

    return sum_all_scores, detailed_crps_data


def score_models(
    model_predictions: dict[str, np.ndarray],
    real_price_path: np.ndarray,
    scoring_config: ScoringConfig,
) -> dict[str, dict]:
    """Score multiple models' predictions against reality.

    Returns raw CRPS scores — no normalisation or softmax.

    Parameters
    ----------
    model_predictions : dict
        model_name -> np.ndarray of shape (num_sims, num_steps).
    real_price_path : np.ndarray
        Shape (num_steps,) — actual prices.
    scoring_config : ScoringConfig
        Which scoring intervals / parameters to use.

    Returns
    -------
    dict of model_name -> {raw_crps, detailed}
    """
    results: dict[str, dict] = {}

    for name, paths in model_predictions.items():
        crps_total, details = calculate_crps_for_miner(
            paths,
            real_price_path,
            scoring_config.time_increment,
            scoring_config.scoring_intervals,
        )
        results[name] = {
            "raw_crps": crps_total,
            "detailed": details,
        }

    return results
