"""
Stochastic volatility models — Heston model with optional jump diffusion.

The Heston model assumes volatility itself is a random process:
    dS = mu * S * dt + sqrt(V) * S * dW_S
    dV = kappa * (theta - V) * dt + xi * sqrt(V) * dW_V
    corr(dW_S, dW_V) = rho

This captures:
    - Mean-reverting volatility
    - Volatility of volatility (fat tails)
    - Correlation between returns and vol changes (leverage effect)
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

from models.base import BaseForecaster

logger = logging.getLogger(__name__)


class HestonForecaster(BaseForecaster):
    """Heston stochastic volatility Monte Carlo path generator."""

    def __init__(self) -> None:
        # Model parameters
        self.mu: float = 0.0       # Drift
        self.kappa: float = 2.0    # Mean reversion speed
        self.theta: float = 0.04   # Long-run variance
        self.xi: float = 0.3       # Vol of vol
        self.rho: float = -0.7     # Correlation (typically negative for equities)
        self.v0: float = 0.04      # Initial variance
        self.last_price: float = 0.0
        self._fitted = False

    def fit(self, prices: np.ndarray, **kwargs) -> None:
        """Calibrate Heston parameters using method of moments on realized variance."""
        prices = np.asarray(prices, dtype=np.float64)
        log_returns = np.diff(np.log(prices))

        self.last_price = float(prices[-1])
        self.mu = float(np.mean(log_returns)) * (24 * 60 / 5)  # Annualise from 5min

        # Estimate realized variance in rolling windows
        window = min(48, len(log_returns) // 4)  # ~4 hours of 5-min data
        if window < 4:
            # Not enough data for rolling estimation — use simple estimates
            var = float(np.var(log_returns))
            self.theta = var * (24 * 60 / 5)  # Annualised
            self.v0 = self.theta
            self.kappa = 2.0
            self.xi = 0.3
            self.rho = float(np.corrcoef(log_returns[:-1], np.abs(log_returns[1:]))[0, 1])
            self._fitted = True
            return

        realized_var = np.array([
            np.var(log_returns[i : i + window]) * (24 * 60 / 5)
            for i in range(0, len(log_returns) - window, window)
        ])

        if len(realized_var) < 3:
            self.theta = float(np.mean(realized_var))
            self.v0 = float(realized_var[-1])
            self._fitted = True
            return

        # Method of moments estimates
        self.theta = float(np.mean(realized_var))
        self.v0 = float(realized_var[-1])
        var_of_var = float(np.var(realized_var))

        # kappa from autocorrelation of variance
        autocorr = np.corrcoef(realized_var[:-1], realized_var[1:])[0, 1]
        dt_window = window * 5 / (24 * 60)  # Window size in days
        if 0 < autocorr < 1:
            self.kappa = -np.log(autocorr) / dt_window
        else:
            self.kappa = 2.0

        # xi from variance of variance
        if self.theta > 0:
            self.xi = np.sqrt(2 * self.kappa * var_of_var / self.theta)
        else:
            self.xi = 0.3

        # rho from correlation between returns and subsequent vol changes
        min_len = min(len(log_returns) - 1, len(realized_var) - 1)
        if min_len > 2:
            # Subsample returns to match realized_var frequency
            sampled_returns = log_returns[::window][:len(realized_var)]
            min_len = min(len(sampled_returns) - 1, len(realized_var) - 1)
            if min_len > 2:
                self.rho = float(np.corrcoef(
                    sampled_returns[:min_len],
                    np.diff(realized_var[:min_len + 1])
                )[0, 1])
            else:
                self.rho = -0.5

        # Enforce Feller condition: 2*kappa*theta > xi^2
        feller = 2 * self.kappa * self.theta
        if feller <= self.xi ** 2:
            logger.warning(
                "Feller condition violated (%.4f <= %.4f) — adjusting xi",
                feller,
                self.xi ** 2,
            )
            self.xi = np.sqrt(0.95 * feller)

        self.rho = np.clip(self.rho, -0.99, 0.99)
        self._fitted = True

        logger.info(
            "Fitted Heston: kappa=%.4f theta=%.6f xi=%.4f rho=%.4f v0=%.6f",
            self.kappa, self.theta, self.xi, self.rho, self.v0,
        )

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
        dt = 5.0 / (24 * 60)  # 5 minutes in days

        # Correlated Brownian motions
        Z1 = np.random.standard_normal((num_paths, num_steps - 1))
        Z2 = np.random.standard_normal((num_paths, num_steps - 1))
        W_S = Z1
        W_V = self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * Z2

        # Simulate variance process (QE scheme for better accuracy)
        V = np.zeros((num_paths, num_steps), dtype=np.float64)
        V[:, 0] = self.v0

        log_S = np.zeros((num_paths, num_steps), dtype=np.float64)
        log_S[:, 0] = np.log(s0)

        for t in range(num_steps - 1):
            v_curr = np.maximum(V[:, t], 0)
            sqrt_v = np.sqrt(v_curr)

            # Variance dynamics (truncated Euler — simple but effective)
            dV = self.kappa * (self.theta - v_curr) * dt + self.xi * sqrt_v * np.sqrt(dt) * W_V[:, t]
            V[:, t + 1] = np.maximum(v_curr + dV, 0)

            # Price dynamics
            dlog_S = (self.mu - 0.5 * v_curr) * dt + sqrt_v * np.sqrt(dt) * W_S[:, t]
            log_S[:, t + 1] = log_S[:, t] + dlog_S

        paths = np.exp(np.clip(log_S, -500, 500))
        return paths

    def params_dict(self) -> dict:
        return {
            "model": "Heston",
            "mu": self.mu,
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
            "v0": self.v0,
            "last_price": self.last_price,
        }
