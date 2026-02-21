"""
Market data tools — fetch live and historical price data for SN50 assets.
"""

from __future__ import annotations

import json
import logging
import time

import httpx
import numpy as np
import pandas as pd

from config import PYTH_PRICE_FEED_URL
from pipeline.tools.registry import tool

logger = logging.getLogger(__name__)

# Pyth price feed IDs for SN50 assets (mainnet)
# These are the canonical Pyth feed IDs — update if Pyth changes them.
PYTH_FEED_IDS: dict[str, str] = {
    "BTC": "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
    "ETH": "ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
    "SOL": "ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
    "XAU": "765d2ba906dbc32ca17cc11f5310a89e9ee1f6420508c63861f2f8ba4ee34bb2",
}


@tool(description="Fetch the latest price for an SN50 asset from Pyth oracle.")
def get_latest_price(asset: str) -> str:
    """Fetch the latest price for the given asset."""
    feed_id = PYTH_FEED_IDS.get(asset.upper())
    if not feed_id:
        return json.dumps({"error": f"No Pyth feed ID for asset: {asset}"})

    url = f"{PYTH_PRICE_FEED_URL}/v2/updates/price/latest"
    try:
        resp = httpx.get(url, params={"ids[]": feed_id}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "parsed" in data and data["parsed"]:
            price_data = data["parsed"][0]["price"]
            price = int(price_data["price"]) * (10 ** int(price_data["expo"]))
            return json.dumps({
                "asset": asset,
                "price": price,
                "confidence": int(price_data["conf"]) * (10 ** int(price_data["expo"])),
                "timestamp": price_data.get("publish_time", int(time.time())),
            })
        return json.dumps({"error": "No price data returned"})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Fetch historical OHLCV data for an asset. Returns JSON"
        " with columns: timestamp, open, high, low, close, volume."
    ),
)
def get_historical_data(asset: str, days: int = 30) -> str:
    """Fetch historical price data.

    Uses public crypto APIs for BTC/ETH/SOL; returns synthetic data for
    tokenized assets if no public API is available.
    """
    try:
        if asset.upper() in ("BTC", "ETH", "SOL"):
            return _fetch_crypto_ohlcv(asset.upper(), days)
        else:
            return json.dumps({
                "asset": asset,
                "note": "Historical data for tokenized assets requires a premium data provider.",
                "suggestion": "Use crypto proxy or configure a data provider in .env",
            })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def _fetch_crypto_ohlcv(symbol: str, days: int) -> str:
    """Fetch OHLCV from CoinGecko (free tier)."""
    cg_ids = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
    cg_id = cg_ids.get(symbol)
    if not cg_id:
        return json.dumps({"error": f"Unknown crypto symbol: {symbol}"})

    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/ohlc"
    resp = httpx.get(url, params={"vs_currency": "usd", "days": str(days)}, timeout=30)
    resp.raise_for_status()
    raw = resp.json()  # list of [timestamp, open, high, low, close]
    records = [
        {"timestamp": r[0] / 1000, "open": r[1], "high": r[2], "low": r[3], "close": r[4]}
        for r in raw
    ]
    return json.dumps({"asset": symbol, "interval": "4h" if days <= 30 else "1d", "data": records})


@tool(description="Compute log returns and basic statistics from historical price data JSON.")
def compute_returns_stats(price_data_json: str) -> str:
    """Given historical data JSON (from get_historical_data), compute log returns and stats."""
    try:
        data = json.loads(price_data_json)
        if "error" in data:
            return price_data_json
        records = data.get("data", [])
        if not records:
            return json.dumps({"error": "No price records"})

        closes = np.array([r["close"] for r in records], dtype=np.float64)
        log_returns = np.diff(np.log(closes))

        stats = {
            "asset": data.get("asset", "unknown"),
            "n_observations": len(closes),
            "n_returns": len(log_returns),
            "mean_return": float(np.mean(log_returns)),
            "std_return": float(np.std(log_returns)),
            "annualized_vol": float(np.std(log_returns) * np.sqrt(365 * 6)),  # ~6 4h candles/day
            "skewness": float(pd.Series(log_returns).skew()),
            "kurtosis": float(pd.Series(log_returns).kurtosis()),
            "min_return": float(np.min(log_returns)),
            "max_return": float(np.max(log_returns)),
            "last_price": float(closes[-1]),
        }
        return json.dumps(stats, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
