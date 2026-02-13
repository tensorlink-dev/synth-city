"""
Data preprocessing utilities for model training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from a price series."""
    prices = np.asarray(prices, dtype=np.float64)
    return np.diff(np.log(prices))


def compute_realized_variance(
    log_returns: np.ndarray,
    window: int = 12,
) -> np.ndarray:
    """Compute rolling realized variance.

    Parameters
    ----------
    log_returns : np.ndarray
        1-D array of log returns.
    window : int
        Rolling window size.

    Returns
    -------
    np.ndarray of same length as log_returns (NaN-padded at start).
    """
    series = pd.Series(log_returns)
    rv = series.rolling(window=window).var()
    return rv.values


def detect_regime(log_returns: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    """Simple regime detection based on rolling volatility.

    Returns an array of regime labels:
        0 = low volatility
        1 = normal volatility
        2 = high volatility
    """
    rv = compute_realized_variance(log_returns, window=24)
    mean_rv = np.nanmean(rv)
    std_rv = np.nanstd(rv)

    regimes = np.zeros(len(log_returns), dtype=int)
    regimes[rv > mean_rv + threshold * std_rv] = 2  # High vol
    regimes[(rv > mean_rv - 0.5 * std_rv) & (rv <= mean_rv + threshold * std_rv)] = 1  # Normal
    return regimes


def prepare_training_features(
    prices: np.ndarray,
    include_volume: bool = False,
    volumes: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Prepare a feature dict for model training.

    Returns
    -------
    dict with keys: log_returns, realized_var_short, realized_var_long, regimes
    """
    log_returns = compute_log_returns(prices)
    rv_short = compute_realized_variance(log_returns, window=12)   # ~1h at 5min
    rv_long = compute_realized_variance(log_returns, window=48)    # ~4h at 5min
    regimes = detect_regime(log_returns)

    features = {
        "log_returns": log_returns,
        "realized_var_short": rv_short,
        "realized_var_long": rv_long,
        "regimes": regimes,
    }

    if include_volume and volumes is not None:
        features["log_volume"] = np.log1p(volumes[1:])  # Align with returns

    return features


def split_train_val(
    data: np.ndarray,
    val_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a time series into train/validation (no shuffle — temporal order)."""
    split_idx = int(len(data) * (1 - val_fraction))
    return data[:split_idx], data[split_idx:]
