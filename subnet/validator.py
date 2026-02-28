"""
CRPS scoring — local implementation for evaluating model quality before
submitting to the network.

CRPS (Continuous Ranked Probability Score) measures how well a set of
simulated paths captures the true distribution of future prices.

    CRPS = (1/N) * SUM(|y_n - x|) - (1/2*N^2) * SUM_n SUM_m (|y_n - y_m|)

Lower CRPS = better calibration.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from config import SN50_ASSETS, SN50_STEP_MINUTES
from subnet.config import CRPS_EVAL_INCREMENTS

logger = logging.getLogger(__name__)


def crps_ensemble(forecasts: np.ndarray, observation: float) -> float:
    """Compute CRPS for an ensemble forecast at a single time step.

    Parameters
    ----------
    forecasts : np.ndarray
        1-D array of N forecast values (one per simulated path).
    observation : float
        The realised value.

    Returns
    -------
    float
        CRPS score (lower is better).
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    N = len(forecasts)
    if N == 0:
        return float("inf")

    # Term 1: mean absolute error
    mae = np.mean(np.abs(forecasts - observation))

    # Term 2: mean pairwise absolute difference (efficient O(N log N) via sorted array)
    sorted_f = np.sort(forecasts)
    # Using the identity: sum |y_i - y_j| = 2 * sum_i (y_i * (2*i - N - 1)) / N^2 for sorted y
    indices = np.arange(1, N + 1)
    pairwise = 2.0 * np.sum(sorted_f * (2 * indices - N - 1)) / (N * N)

    return mae - 0.5 * pairwise


def crps_basis_points(forecasts: np.ndarray, observation: float, reference_price: float) -> float:
    """CRPS measured in basis points relative to a reference price.

    This normalizes scores across assets with different price scales.
    """
    if reference_price <= 0:
        return float("inf")
    # Convert to basis points (1 bp = 0.01%)
    forecasts_bp = (forecasts / reference_price - 1) * 10_000
    observation_bp = (observation / reference_price - 1) * 10_000
    return crps_ensemble(forecasts_bp, observation_bp)


def evaluate_prediction(
    paths: np.ndarray,
    realized_prices: np.ndarray,
    step_minutes: int = SN50_STEP_MINUTES,
) -> dict[str, float]:
    """Evaluate a full prediction against realised prices.

    Parameters
    ----------
    paths : np.ndarray
        Shape (num_paths, num_steps) — the Monte Carlo prediction.
    realized_prices : np.ndarray
        Shape (num_steps,) — the actual price path that occurred.
    step_minutes : int
        Minutes between time steps.

    Returns
    -------
    dict with CRPS scores at various time horizons.
    """
    num_paths, num_steps = paths.shape
    assert len(realized_prices) >= num_steps, "Not enough realised data"

    reference_price = realized_prices[0]  # t0 price
    scores = {}

    # Score at each evaluation increment
    for minutes in CRPS_EVAL_INCREMENTS:
        step_idx = minutes // step_minutes
        if step_idx >= num_steps:
            continue
        score = crps_basis_points(
            paths[:, step_idx],
            realized_prices[step_idx],
            reference_price,
        )
        scores[f"crps_{minutes}min"] = score

    # Final price component
    scores["crps_final"] = crps_basis_points(
        paths[:, -1],
        realized_prices[num_steps - 1],
        reference_price,
    )

    # Aggregate CRPS sum
    scores["crps_sum"] = sum(scores.values())

    return scores


def evaluate_multi_asset(
    predictions: dict[str, np.ndarray],
    realized: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Evaluate predictions across all SN50 assets with proper weighting.

    Parameters
    ----------
    predictions : dict
        asset -> paths array (num_paths, num_steps)
    realized : dict
        asset -> realized_prices array (num_steps,)

    Returns
    -------
    dict with per-asset scores, weighted total, and metadata.
    """
    per_asset: dict[str, dict[str, Any]] = {}
    weighted_sum = 0.0

    for asset, weight in SN50_ASSETS.items():
        if asset in predictions and asset in realized:
            scores = evaluate_prediction(predictions[asset], realized[asset])
            per_asset[asset] = scores
            weighted_sum += scores["crps_sum"] * weight
        else:
            # Missing prediction — assign penalty (90th percentile placeholder)
            per_asset[asset] = {"crps_sum": float("inf"), "status": "missing"}

    return {
        "per_asset": per_asset,
        "weighted_crps_sum": weighted_sum,
        "assets_predicted": len(predictions),
        "assets_total": len(SN50_ASSETS),
    }
