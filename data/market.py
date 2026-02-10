"""
Market data fetching and caching — provides historical and live price data
for model fitting and evaluation.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from config import PYTH_PRICE_FEED_URL, SN50_ASSETS, WORKSPACE_DIR

logger = logging.getLogger(__name__)

# CoinGecko free-tier mapping for crypto assets
_COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}

# Cache directory
CACHE_DIR = WORKSPACE_DIR / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ohlcv(
    asset: str,
    days: int = 30,
    use_cache: bool = True,
    cache_ttl_hours: int = 1,
) -> pd.DataFrame:
    """Fetch OHLCV data for an asset.

    Parameters
    ----------
    asset : str
        Asset symbol (e.g. "BTC", "ETH").
    days : int
        Number of days of history.
    use_cache : bool
        Whether to use/update the file cache.
    cache_ttl_hours : int
        Cache expiry in hours.

    Returns
    -------
    pd.DataFrame with columns: timestamp, open, high, low, close
    """
    cache_path = CACHE_DIR / f"{asset}_{days}d_ohlcv.parquet"

    # Check cache
    if use_cache and cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < cache_ttl_hours:
            logger.debug("Using cached data for %s (%d days, %.1fh old)", asset, days, age_hours)
            return pd.read_parquet(cache_path)

    # Fetch from API
    cg_id = _COINGECKO_IDS.get(asset.upper())
    if cg_id:
        df = _fetch_coingecko_ohlcv(cg_id, days)
    else:
        logger.warning("No data source for %s — returning empty DataFrame", asset)
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])

    # Cache
    if use_cache and not df.empty:
        df.to_parquet(cache_path, index=False)
        logger.debug("Cached %d rows for %s", len(df), asset)

    return df


def _fetch_coingecko_ohlcv(cg_id: str, days: int) -> pd.DataFrame:
    """Fetch OHLCV from CoinGecko free API."""
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/ohlc"
    resp = httpx.get(url, params={"vs_currency": "usd", "days": str(days)}, timeout=30)
    resp.raise_for_status()
    raw = resp.json()  # list of [timestamp_ms, open, high, low, close]

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.sort_values("timestamp").reset_index(drop=True)


def get_close_prices(asset: str, days: int = 30) -> np.ndarray:
    """Return a 1-D numpy array of close prices for model fitting."""
    df = fetch_ohlcv(asset, days)
    if df.empty:
        raise ValueError(f"No data available for {asset}")
    return df["close"].values.astype(np.float64)


def get_latest_prices(assets: list[str] | None = None) -> dict[str, float]:
    """Fetch the latest price for each asset.

    Uses the most recent close from OHLCV data as a fallback when Pyth
    is unavailable.
    """
    assets = assets or list(SN50_ASSETS.keys())
    prices = {}

    for asset in assets:
        try:
            df = fetch_ohlcv(asset, days=1, cache_ttl_hours=0)
            if not df.empty:
                prices[asset] = float(df["close"].iloc[-1])
        except Exception as exc:
            logger.warning("Failed to fetch price for %s: %s", asset, exc)

    return prices
