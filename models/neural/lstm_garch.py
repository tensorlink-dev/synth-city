"""
LSTM-GARCH hybrid — embeds GARCH structure into a neural network for
non-linear volatility dynamics.

Architecture:
    1. LSTM encodes a sequence of (log_return, realized_vol, features) into
       a hidden state.
    2. A linear head predicts GARCH-like parameters (omega, alpha, beta)
       conditioned on the hidden state.
    3. Monte Carlo paths are generated using the predicted time-varying
       GARCH parameters.

This captures the GARCH inductive bias (volatility clustering) while
allowing the LSTM to learn non-linear regime transitions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.base import BaseForecaster

logger = logging.getLogger(__name__)


class LSTMGARCHNet(nn.Module):
    """Neural network that predicts GARCH parameters from a return sequence."""

    def __init__(self, input_size: int = 3, hidden_size: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )
        # Predict: omega, alpha, beta, mu, log_nu (t-distribution df)
        self.head = nn.Linear(hidden_size, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, input_size)

        Returns
        -------
        Tensor of shape (batch, 5) — [omega, alpha, beta, mu, log_nu]
        """
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]  # Take last time step
        params = self.head(last_hidden)

        # Enforce parameter constraints via activations
        omega = torch.softplus(params[:, 0:1]) * 1e-4   # Small positive
        alpha = torch.sigmoid(params[:, 1:2]) * 0.5     # [0, 0.5]
        beta = torch.sigmoid(params[:, 2:3]) * 0.99     # [0, 0.99]
        mu = params[:, 3:4] * 0.01                       # Small drift
        log_nu = torch.clamp(params[:, 4:5], min=0.7, max=4.0)  # nu in [2, ~55]

        return torch.cat([omega, alpha, beta, mu, log_nu], dim=1)


class LSTMGARCHForecaster(BaseForecaster):
    """LSTM-GARCH hybrid forecaster.

    The network is trained on historical data to predict GARCH parameters,
    then those parameters drive Monte Carlo simulation.
    """

    def __init__(self, hidden_size: int = 64, seq_len: int = 96) -> None:
        self.hidden_size = hidden_size
        self.seq_len = seq_len  # Number of historical steps to condition on
        self.net = LSTMGARCHNet(input_size=3, hidden_size=hidden_size)
        self.last_price: float = 0.0
        self.last_sequence: np.ndarray | None = None
        self._fitted = False

    def fit(self, prices: np.ndarray, epochs: int = 100, lr: float = 1e-3, **kwargs) -> None:
        """Train the LSTM-GARCH network on historical data.

        Training objective: predict next-step realized variance using
        GARCH-parameterised likelihood.
        """
        prices = np.asarray(prices, dtype=np.float64)
        log_returns = np.diff(np.log(prices))

        # Compute features: (log_return, realized_vol_short, realized_vol_long)
        short_window = 12  # 1 hour of 5-min data
        long_window = 48   # 4 hours
        realized_vol_short = np.array([
            np.std(log_returns[max(0, i - short_window):i + 1])
            for i in range(len(log_returns))
        ])
        realized_vol_long = np.array([
            np.std(log_returns[max(0, i - long_window):i + 1])
            for i in range(len(log_returns))
        ])

        features = np.stack([log_returns, realized_vol_short, realized_vol_long], axis=1)

        # Build training sequences
        X, Y = [], []
        for i in range(self.seq_len, len(features) - 1):
            X.append(features[i - self.seq_len : i])
            # Target: next-step squared return (proxy for variance)
            Y.append(log_returns[i + 1] ** 2)

        if not X:
            logger.warning("Not enough data for LSTM-GARCH training")
            self.last_price = float(prices[-1])
            self._fitted = False
            return

        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        Y_tensor = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(1)

        # Train
        self.net.train()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            params = self.net(X_tensor)

            omega = params[:, 0:1]
            alpha = params[:, 1:2]
            beta = params[:, 2:3]

            # GARCH(1,1) variance: sigma^2 = omega + alpha * y^2_{t-1} + beta * sigma^2_{t-1}
            # Simplified: use unconditional variance as approximation for training
            pred_var = omega / (1 - alpha - beta + 1e-8)

            # MSE loss on variance prediction
            loss = nn.functional.mse_loss(pred_var, Y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                logger.debug("Epoch %d/%d loss=%.6f", epoch + 1, epochs, loss.item())

        self.net.eval()
        self.last_price = float(prices[-1])
        self.last_sequence = features[-self.seq_len:]
        self._fitted = True

    def generate_paths(
        self,
        asset: str,
        num_paths: int,
        num_steps: int,
        s0: float | None = None,
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        s0 = s0 or self.last_price

        # Get GARCH parameters from the network
        with torch.no_grad():
            seq_tensor = torch.tensor(self.last_sequence, dtype=torch.float32).unsqueeze(0)
            params = self.net(seq_tensor).squeeze(0).numpy()

        omega, alpha, beta, mu, log_nu = params
        nu = np.exp(log_nu) + 2  # Ensure nu > 2

        # Generate paths using predicted GARCH parameters
        dt = 5.0 / (24 * 60)

        # t-distributed innovations
        if nu > 2:
            innovations = np.random.standard_t(df=nu, size=(num_paths, num_steps - 1))
            innovations = innovations / np.sqrt(nu / (nu - 2))
        else:
            innovations = np.random.standard_normal((num_paths, num_steps - 1))

        # GARCH variance simulation
        variances = np.zeros((num_paths, num_steps - 1), dtype=np.float64)
        returns = np.zeros((num_paths, num_steps - 1), dtype=np.float64)

        unconditional_var = omega / max(1 - alpha - beta, 1e-8)
        prev_var = np.full(num_paths, unconditional_var, dtype=np.float64)
        prev_resid = np.zeros(num_paths, dtype=np.float64)

        for t in range(num_steps - 1):
            curr_var = omega + alpha * prev_resid ** 2 + beta * prev_var
            curr_var = np.maximum(curr_var, 1e-12)
            variances[:, t] = curr_var

            sigma = np.sqrt(curr_var)
            resid = sigma * innovations[:, t]
            returns[:, t] = mu * dt + resid * np.sqrt(dt)

            prev_var = curr_var
            prev_resid = resid

        # Convert to price paths
        cumulative_returns = np.cumsum(returns, axis=1)
        paths = np.zeros((num_paths, num_steps), dtype=np.float64)
        paths[:, 0] = s0
        paths[:, 1:] = s0 * np.exp(cumulative_returns)

        return paths

    def params_dict(self) -> dict:
        return {
            "model": "LSTM-GARCH",
            "hidden_size": self.hidden_size,
            "seq_len": self.seq_len,
            "last_price": self.last_price,
            "fitted": self._fitted,
        }

    def save(self, path: str | Path) -> None:
        """Save the network weights."""
        torch.save(self.net.state_dict(), path)

    def load(self, path: str | Path) -> None:
        """Load network weights."""
        self.net.load_state_dict(torch.load(path, weights_only=True))
        self.net.eval()
