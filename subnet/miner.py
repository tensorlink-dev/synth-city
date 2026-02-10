"""
SN50 Miner — generates and submits probabilistic price forecasts.

The miner:
    1. Receives prediction requests from validators.
    2. Fetches the latest price for each asset.
    3. Generates 1,000 Monte Carlo paths using the best available model.
    4. Formats and returns the prediction payload.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np

from config import SN50_ASSETS, SN50_NUM_PATHS
from models.base import BaseForecaster
from subnet.config import NUM_STEPS_24H, NUM_STEPS_HFT

logger = logging.getLogger(__name__)


class SynthMiner:
    """Miner for Bittensor Subnet 50 (Synth).

    Manages a registry of fitted models per asset and handles prediction
    request formatting.
    """

    def __init__(self) -> None:
        # asset -> fitted model instance
        self.models: dict[str, BaseForecaster] = {}
        # asset -> latest price
        self.prices: dict[str, float] = {}

    def register_model(self, asset: str, model: BaseForecaster) -> None:
        """Register a fitted model for an asset."""
        self.models[asset] = model
        logger.info("Registered model for %s: %s", asset, type(model).__name__)

    def set_price(self, asset: str, price: float) -> None:
        """Update the latest price for an asset."""
        self.prices[asset] = price

    def generate_prediction(
        self,
        asset: str,
        horizon: str = "24h",
    ) -> dict[str, Any]:
        """Generate a prediction payload for one asset.

        Parameters
        ----------
        asset : str
            Asset symbol.
        horizon : str
            "24h" for the standard forecast or "1h" for HFT.

        Returns
        -------
        dict with keys: asset, t0, horizon, num_paths, num_steps, paths
        """
        model = self.models.get(asset)
        if model is None:
            raise ValueError(f"No model registered for asset: {asset}")

        s0 = self.prices.get(asset)
        num_steps = NUM_STEPS_24H if horizon == "24h" else NUM_STEPS_HFT
        t0 = time.time()

        paths = model.generate_paths(
            asset=asset,
            num_paths=SN50_NUM_PATHS,
            num_steps=num_steps,
            s0=s0,
        )

        # Validate output
        assert paths.shape == (SN50_NUM_PATHS, num_steps), (
            f"Shape mismatch: got {paths.shape}, expected ({SN50_NUM_PATHS}, {num_steps})"
        )
        assert not np.isnan(paths).any(), "NaN detected in paths"
        assert not np.isinf(paths).any(), "Inf detected in paths"
        assert (paths > 0).all(), "Non-positive prices detected"

        return {
            "asset": asset,
            "t0": t0,
            "horizon": horizon,
            "num_paths": SN50_NUM_PATHS,
            "num_steps": num_steps,
            "step_minutes": 5,
            "paths": paths.tolist(),
        }

    def generate_all_predictions(self, horizon: str = "24h") -> list[dict[str, Any]]:
        """Generate predictions for all registered assets."""
        predictions = []
        for asset in SN50_ASSETS:
            if asset not in self.models:
                logger.warning("No model for %s — will receive 90th percentile penalty", asset)
                continue
            try:
                pred = self.generate_prediction(asset, horizon=horizon)
                predictions.append(pred)
                logger.info(
                    "Generated %s prediction for %s: %d paths x %d steps",
                    horizon, asset, pred["num_paths"], pred["num_steps"],
                )
            except Exception as exc:
                logger.error("Failed to generate prediction for %s: %s", asset, exc)
        return predictions

    def format_submission(self, predictions: list[dict[str, Any]]) -> dict[str, Any]:
        """Format predictions into the SN50 submission payload."""
        return {
            "miner_id": "synth-city",
            "timestamp": time.time(),
            "predictions": {
                p["asset"]: {
                    "t0": p["t0"],
                    "horizon": p["horizon"],
                    "step_minutes": p["step_minutes"],
                    "num_paths": p["num_paths"],
                    "num_steps": p["num_steps"],
                    # Paths are transmitted as nested lists
                    "paths": p["paths"],
                }
                for p in predictions
            },
        }
