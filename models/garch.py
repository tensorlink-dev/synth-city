"""
GARCH family models — capture volatility clustering for better CRPS scores.

Implements:
    - GARCH(1,1)
    - EGARCH(1,1) — captures leverage effects (asymmetric vol response)
    - GJR-GARCH(1,1) — another asymmetric variant

Uses the ``arch`` library for parameter estimation, then generates Monte
Carlo paths with time-varying volatility.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

import numpy as np

from models.base import BaseForecaster

logger = logging.getLogger(__name__)


class GARCHForecaster(BaseForecaster):
    """GARCH-family Monte Carlo path generator.

    Parameters
    ----------
    variant : str
        One of "GARCH", "EGARCH", "GJR-GARCH".
    p, q : int
        GARCH order parameters.
    dist : str
        Innovation distribution: "normal", "t", "skewt".
    """

    def __init__(
        self,
        variant: Literal["GARCH", "EGARCH", "GJR-GARCH"] = "GARCH",
        p: int = 1,
        q: int = 1,
        dist: str = "t",
    ) -> None:
        self.variant = variant
        self.p = p
        self.q = q
        self.dist = dist

        # Fitted parameters (populated by fit())
        self.omega: float = 0.0
        self.alpha: list[float] = []
        self.beta: list[float] = []
        self.gamma: list[float] = []  # For asymmetric models
        self.mu: float = 0.0
        self.nu: float = 5.0  # Degrees of freedom for t-dist
        self.skew: float = 0.0
        self.last_price: float = 0.0
        self.last_variance: float = 0.0
        self.last_resid: float = 0.0
        self._fitted = False

    def fit(self, prices: np.ndarray, **kwargs) -> None:
        """Fit GARCH parameters using the arch library."""
        from arch import arch_model

        prices = np.asarray(prices, dtype=np.float64)
        # Work with percentage log returns for numerical stability
        log_returns = np.diff(np.log(prices)) * 100

        vol_model = self.variant.replace("-", "")  # "GJRGARCH" -> arch uses "GARCH" + o param
        if self.variant == "GJR-GARCH":
            model = arch_model(
                log_returns, mean="Constant", vol="GARCH", p=self.p, o=1, q=self.q, dist=self.dist
            )
        elif self.variant == "EGARCH":
            model = arch_model(
                log_returns, mean="Constant", vol="EGARCH", p=self.p, q=self.q, dist=self.dist
            )
        else:
            model = arch_model(
                log_returns, mean="Constant", vol="GARCH", p=self.p, q=self.q, dist=self.dist
            )

        result = model.fit(disp="off", show_warning=False)
        params = result.params

        self.mu = float(params.get("mu", 0.0))
        self.omega = float(params.get("omega", 0.0))

        # Extract alpha, beta, gamma
        self.alpha = [float(params.get(f"alpha[{i+1}]", 0.0)) for i in range(self.p)]
        self.beta = [float(params.get(f"beta[{i+1}]", 0.0)) for i in range(self.q)]

        if self.variant == "GJR-GARCH":
            self.gamma = [float(params.get("gamma[1]", 0.0))]
        elif self.variant == "EGARCH":
            self.gamma = [float(params.get(f"gamma[{i+1}]", 0.0)) for i in range(self.p)]

        if self.dist in ("t", "skewt"):
            self.nu = float(params.get("nu", 5.0))
        if self.dist == "skewt":
            self.skew = float(params.get("lambda", 0.0))

        # Store state for simulation warm-start
        self.last_price = float(prices[-1])
        cond_var = result.conditional_volatility ** 2
        self.last_variance = float(cond_var[-1]) if len(cond_var) > 0 else self.omega
        resids = result.resid
        self.last_resid = float(resids[-1]) if len(resids) > 0 else 0.0
        self._fitted = True

        logger.info(
            "Fitted %s: omega=%.6f alpha=%s beta=%s gamma=%s nu=%.2f",
            self.variant, self.omega, self.alpha, self.beta, self.gamma, self.nu,
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

        # Generate innovations from the fitted distribution
        if self.dist == "t" and self.nu > 2:
            innovations = np.random.standard_t(df=self.nu, size=(num_paths, num_steps - 1))
            # Scale to unit variance
            innovations = innovations / np.sqrt(self.nu / (self.nu - 2))
        else:
            innovations = np.random.standard_normal((num_paths, num_steps - 1))

        # Simulate GARCH variance process
        variances = np.zeros((num_paths, num_steps - 1), dtype=np.float64)
        returns = np.zeros((num_paths, num_steps - 1), dtype=np.float64)

        # Initialize with last fitted values
        prev_var = np.full(num_paths, self.last_variance, dtype=np.float64)
        prev_resid = np.full(num_paths, self.last_resid, dtype=np.float64)

        alpha_sum = self.alpha[0] if self.alpha else 0.0
        beta_sum = self.beta[0] if self.beta else 0.0
        gamma_val = self.gamma[0] if self.gamma else 0.0

        for t in range(num_steps - 1):
            if self.variant == "EGARCH":
                # EGARCH: log(sigma^2_t) = omega + alpha * g(z) + beta * log(sigma^2_{t-1})
                log_var = (
                    self.omega
                    + alpha_sum * (np.abs(prev_resid / np.sqrt(np.maximum(prev_var, 1e-12)))
                                   - np.sqrt(2 / np.pi))
                    + gamma_val * (prev_resid / np.sqrt(np.maximum(prev_var, 1e-12)))
                    + beta_sum * np.log(np.maximum(prev_var, 1e-12))
                )
                curr_var = np.exp(np.clip(log_var, -20, 20))
            elif self.variant == "GJR-GARCH":
                indicator = (prev_resid < 0).astype(np.float64)
                curr_var = (
                    self.omega
                    + alpha_sum * prev_resid ** 2
                    + gamma_val * indicator * prev_resid ** 2
                    + beta_sum * prev_var
                )
            else:
                # Standard GARCH
                curr_var = (
                    self.omega
                    + alpha_sum * prev_resid ** 2
                    + beta_sum * prev_var
                )

            # Floor variance to prevent numerical issues
            curr_var = np.maximum(curr_var, 1e-12)
            variances[:, t] = curr_var

            # Generate returns (percentage log returns)
            sigma = np.sqrt(curr_var)
            resid = sigma * innovations[:, t]
            returns[:, t] = self.mu + resid

            prev_var = curr_var
            prev_resid = resid

        # Convert percentage log returns to price paths
        cumulative_log_returns = np.cumsum(returns / 100, axis=1)
        paths = np.zeros((num_paths, num_steps), dtype=np.float64)
        paths[:, 0] = s0
        paths[:, 1:] = s0 * np.exp(cumulative_log_returns)

        return paths

    def params_dict(self) -> dict:
        return {
            "model": self.variant,
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "mu": self.mu,
            "nu": self.nu,
            "dist": self.dist,
            "last_price": self.last_price,
        }
