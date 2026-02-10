"""
Base interface for all SN50 forecasting models.

Every model must implement ``generate_paths(asset, num_paths, num_steps)``
returning an ndarray of shape ``(num_paths, num_steps)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseForecaster(ABC):
    """Abstract base class for Monte Carlo path generators."""

    @abstractmethod
    def fit(self, prices: np.ndarray, **kwargs) -> None:
        """Fit model parameters from historical price data.

        Parameters
        ----------
        prices : np.ndarray
            1-D array of historical prices (oldest first).
        """

    @abstractmethod
    def generate_paths(
        self,
        asset: str,
        num_paths: int,
        num_steps: int,
        s0: float | None = None,
    ) -> np.ndarray:
        """Generate Monte Carlo price paths.

        Parameters
        ----------
        asset : str
            Asset symbol (e.g. "BTC").
        num_paths : int
            Number of simulated paths (1000 for SN50).
        num_steps : int
            Number of time steps including t0 (289 for 24h @ 5min).
        s0 : float | None
            Starting price.  If None, use the last price from fit().

        Returns
        -------
        np.ndarray
            Shape ``(num_paths, num_steps)`` — each row is one simulated path.
        """

    def params_dict(self) -> dict:
        """Return fitted parameters as a serialisable dict."""
        return {}
