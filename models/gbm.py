"""
Geometric Brownian Motion — the SN50 baseline model.

GBM assumes constant drift and volatility:
    dS = mu * S * dt + sigma * S * dW

This is the simplest model and serves as a reference.  It will be beaten
by models that capture volatility clustering and fat tails, but it's a
useful starting point and sanity check.
"""

from __future__ import annotations

import numpy as np

from models.base import BaseForecaster


class GBMForecaster(BaseForecaster):
    """Geometric Brownian Motion path generator."""

    def __init__(self) -> None:
        self.mu: float = 0.0
        self.sigma: float = 0.0
        self.last_price: float = 0.0
        self._fitted = False

    def fit(self, prices: np.ndarray, dt: float = 5.0 / (24 * 60), **kwargs) -> None:
        """Estimate mu and sigma from historical prices.

        Parameters
        ----------
        prices : np.ndarray
            Historical prices (oldest first).
        dt : float
            Time step between observations in units of days.
            Default is 5 minutes = 5/(24*60) days.
        """
        prices = np.asarray(prices, dtype=np.float64)
        log_returns = np.diff(np.log(prices))

        self.mu = float(np.mean(log_returns) / dt)
        self.sigma = float(np.std(log_returns) / np.sqrt(dt))
        self.last_price = float(prices[-1])
        self._fitted = True

    def generate_paths(
        self,
        asset: str,
        num_paths: int,
        num_steps: int,
        s0: float | None = None,
    ) -> np.ndarray:
        if not self._fitted and s0 is None:
            raise RuntimeError("Model not fitted and no s0 provided")

        s0 = s0 or self.last_price
        dt = 5.0 / (24 * 60)  # 5-minute steps in days

        # Pre-compute drift and diffusion
        drift = (self.mu - 0.5 * self.sigma ** 2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        # Generate random innovations
        Z = np.random.standard_normal((num_paths, num_steps - 1))

        # Build log-price increments
        log_increments = drift + diffusion * Z

        # Cumulative sum to get log(S_t / S_0)
        log_paths = np.zeros((num_paths, num_steps), dtype=np.float64)
        log_paths[:, 1:] = np.cumsum(log_increments, axis=1)

        # Convert to price levels
        paths = s0 * np.exp(log_paths)

        return paths

    def params_dict(self) -> dict:
        return {
            "model": "GBM",
            "mu": self.mu,
            "sigma": self.sigma,
            "last_price": self.last_price,
        }
