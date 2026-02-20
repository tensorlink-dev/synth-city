"""
Pyth-based live price feed — matches the real SN50 validator's price data provider.

Fetches live prices from the Hermes Pyth oracle and records a price history
for post-hoc CRPS evaluation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
import numpy as np

logger = logging.getLogger(__name__)

# Pyth Hermes price feed IDs (same as synth-subnet)
PYTH_FEED_IDS: dict[str, str] = {
    "BTC": "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
    "ETH": "ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
    "XAU": "765d2ba906dbc32ca17cc11f5310a89e9ee1f6420508c63861f2f8ba4ee34bb2",
    "SOL": "ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
    "SPYX": "2817b78438c769357182c04346fddaad1178c82f4048828fe0997c3c64624e14",
    "NVDAX": "4244d07890e4610f46bbde67de8f43a4bf8b569eebe904f136b469f148503b7f",
    "TSLAX": "47a156470288850a440df3a6ce85a55917b813a19bb5b31128a33a986566a362",
    "AAPLX": "978e6cc68a119ce066aa830017318563a9ed04ec3a0a6439010fc11296a58675",
    "GOOGLX": "b911b0329028cd0283e4259c33809d62942bd2716a58084e5f31d64c00b5424e",
}

PYTH_BASE_URL = "https://hermes.pyth.network/v2/updates/price/latest"


@dataclass
class PriceRecord:
    """A single timestamped price observation."""
    asset: str
    price: float
    timestamp: float  # Unix epoch seconds
    dt: datetime = field(init=False)

    def __post_init__(self) -> None:
        self.dt = datetime.fromtimestamp(self.timestamp, tz=timezone.utc)


class PriceFeed:
    """Live price feed from Pyth Hermes oracle.

    Fetches current prices and maintains a rolling history for CRPS evaluation.
    """

    def __init__(self, assets: list[str] | None = None, max_history: int = 500) -> None:
        self.assets = assets or list(PYTH_FEED_IDS.keys())
        self.max_history = max_history
        # asset -> list of PriceRecord, ordered by time
        self.history: dict[str, list[PriceRecord]] = {a: [] for a in self.assets}
        self._client = httpx.Client(timeout=15)

    def fetch_price(self, asset: str) -> float | None:
        """Fetch the current live price for a single asset from Pyth."""
        feed_id = PYTH_FEED_IDS.get(asset)
        if not feed_id:
            logger.warning("No Pyth feed ID for asset: %s", asset)
            return None

        for attempt in range(3):
            try:
                resp = self._client.get(PYTH_BASE_URL, params={"ids[]": [feed_id]})
                resp.raise_for_status()
                data = resp.json()
                parsed = data.get("parsed", [])
                if not parsed:
                    logger.warning("Empty parsed data from Pyth for %s", asset)
                    return None
                price_info = parsed[0]["price"]
                price = int(price_info["price"]) * (10 ** int(price_info["expo"]))
                return float(price)
            except Exception as exc:
                logger.warning("Pyth fetch attempt %d failed for %s: %s", attempt + 1, asset, exc)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return None

    def fetch_all_prices(self) -> dict[str, float]:
        """Fetch current prices for all tracked assets."""
        prices: dict[str, float] = {}
        for asset in self.assets:
            price = self.fetch_price(asset)
            if price is not None:
                prices[asset] = price
        return prices

    def record_prices(self) -> dict[str, float]:
        """Fetch all prices and append to history. Returns the prices fetched."""
        prices = self.fetch_all_prices()
        ts = time.time()
        for asset, price in prices.items():
            record = PriceRecord(asset=asset, price=price, timestamp=ts)
            self.history[asset].append(record)
            # Trim history
            if len(self.history[asset]) > self.max_history:
                self.history[asset] = self.history[asset][-self.max_history:]
        return prices

    def get_price_path(
        self,
        asset: str,
        start_time: float,
        time_increment: int,
        num_steps: int,
    ) -> np.ndarray | None:
        """Build a real price path from recorded history.

        Interpolates history to match the exact time grid the models predicted on.
        Returns None if insufficient data.

        Parameters
        ----------
        asset : str
            Asset symbol.
        start_time : float
            Unix timestamp of prediction start.
        time_increment : int
            Seconds between steps.
        num_steps : int
            Number of steps (including t0).
        """
        records = self.history.get(asset, [])
        if len(records) < 2:
            return None

        # Build target time grid
        target_times = np.array([start_time + i * time_increment for i in range(num_steps)])

        # Extract recorded times and prices
        rec_times = np.array([r.timestamp for r in records])
        rec_prices = np.array([r.price for r in records])

        # Check coverage: we need data spanning the target range
        if rec_times[-1] < target_times[-1]:
            logger.debug(
                "Insufficient price history for %s: have up to %.0f, need %.0f",
                asset, rec_times[-1], target_times[-1],
            )
            return None

        # Interpolate to target grid
        real_prices = np.interp(target_times, rec_times, rec_prices)
        return real_prices

    def get_historical_prices(self, asset: str, count: int = 100) -> np.ndarray | None:
        """Return the last `count` recorded prices for model fitting."""
        records = self.history.get(asset, [])
        if len(records) < count:
            return None
        prices = np.array([r.price for r in records[-count:]])
        return prices

    def close(self) -> None:
        self._client.close()
