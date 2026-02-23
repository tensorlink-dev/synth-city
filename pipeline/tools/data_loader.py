"""
Data-loader tools — build MarketDataLoader pipelines backed by HuggingFace
parquet datasets for real market data training.

The loaders use open-synth-miner's ``MarketDataLoader`` with ``HFOHLCVSource``
to stream OHLCV candles from ``tensorlink-dev/open-synth-training-data``.

Two timeframes are supported:
  - **5m** — 288-step prediction horizon (24 h at 5-min intervals)
  - **1m** — 60-step prediction horizon (1 h at 1-min intervals)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from config import (
    DATA_BATCH_SIZE,
    DATA_FEATURE_ENGINEER,
    DATA_GAP_HANDLING,
    DATA_STRIDE,
    HF_TRAINING_DATA_REPO,
    RESEARCH_FEATURE_DIM,
    SN50_TO_HF_ASSET,
    TIMEFRAME_CONFIGS,
)
from pipeline.tools.registry import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache: one loader per timeframe
# ---------------------------------------------------------------------------
_loaders: dict[str, Any] = {}


def _build_asset_files(timeframe: str, assets: list[str] | None = None) -> dict[str, str]:
    """Map SN50 asset names to HF parquet paths for *timeframe*."""
    tf_cfg = TIMEFRAME_CONFIGS[timeframe]
    suffix = tf_cfg["file_suffix"]

    mapping: dict[str, str] = {}
    for sn50_name, hf_name in SN50_TO_HF_ASSET.items():
        if assets and sn50_name not in assets:
            continue
        mapping[hf_name] = f"data/{hf_name}/{suffix}"

    return mapping


def _resolve_engineer(name: str) -> Any:
    """Instantiate a FeatureEngineer by short name."""
    from osa.data.market_data_loader import (  # type: ignore[import-untyped]
        WaveletEngineer,
        ZScoreEngineer,
    )

    engineers = {
        "zscore": ZScoreEngineer,
        "wavelet": WaveletEngineer,
    }
    cls = engineers.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown feature engineer {name!r}. Choose from: {list(engineers)}"
        )
    return cls()


def get_loader(
    timeframe: str = "5m",
    assets: list[str] | None = None,
    engineer: str | None = None,
    batch_size: int | None = None,
    stride: int | None = None,
    force_new: bool = False,
) -> Any:
    """Return (or create) a cached ``MarketDataLoader`` for *timeframe*.

    Parameters
    ----------
    timeframe : ``"5m"`` or ``"1m"``
    assets : restrict to these SN50 asset names; ``None`` → all mapped assets
    engineer : feature engineer name (``"zscore"`` or ``"wavelet"``); defaults to
        the ``DATA_FEATURE_ENGINEER`` env var
    batch_size : override ``DATA_BATCH_SIZE``
    stride : override ``DATA_STRIDE``
    force_new : discard any cached loader and rebuild
    """
    from osa.data.market_data_loader import (  # type: ignore[import-untyped]
        HFOHLCVSource,
        MarketDataLoader,
    )

    if timeframe not in TIMEFRAME_CONFIGS:
        raise ValueError(
            f"Unknown timeframe {timeframe!r}. Choose from: {list(TIMEFRAME_CONFIGS)}"
        )

    eng_name = (engineer or DATA_FEATURE_ENGINEER).lower()
    cache_key = f"{timeframe}:{eng_name}:{','.join(sorted(assets)) if assets else 'all'}"
    if not force_new and cache_key in _loaders:
        return _loaders[cache_key]

    tf_cfg = TIMEFRAME_CONFIGS[timeframe]
    asset_files = _build_asset_files(timeframe, assets)
    eng = _resolve_engineer(engineer or DATA_FEATURE_ENGINEER)

    source = HFOHLCVSource(
        repo_id=HF_TRAINING_DATA_REPO,
        asset_files=asset_files,
        repo_type="dataset",
    )

    loader = MarketDataLoader(
        data_source=source,
        engineer=eng,
        assets=list(asset_files.keys()),
        input_len=int(tf_cfg["input_len"]),
        pred_len=int(tf_cfg["pred_len"]),
        batch_size=batch_size or DATA_BATCH_SIZE,
        feature_dim=RESEARCH_FEATURE_DIM,
        gap_handling=DATA_GAP_HANDLING,
        stride=stride or DATA_STRIDE,
    )

    _loaders[cache_key] = loader
    logger.info(
        "Created MarketDataLoader: timeframe=%s, assets=%s, pred_len=%s",
        timeframe,
        list(asset_files.keys()),
        tf_cfg["pred_len"],
    )
    return loader


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------


@tool(
    description=(
        "Create a MarketDataLoader backed by real HF OHLCV data. "
        "timeframe: '5m' (288-step / 24h) or '1m' (60-step / 1h). "
        "engineer: 'zscore' or 'wavelet'. "
        "assets: optional JSON list of SN50 asset names to include. "
        "Returns loader summary with asset list and data shapes."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "timeframe": {
                "type": "string",
                "enum": ["5m", "1m"],
                "description": "Candle interval: '5m' (pred_len=288) or '1m' (pred_len=60)",
            },
            "engineer": {
                "type": "string",
                "enum": ["zscore", "wavelet"],
                "description": "Feature engineer (default: zscore)",
            },
            "assets": {
                "type": "string",
                "description": "JSON array of SN50 asset names, e.g. '[\"BTC\",\"ETH\"]'. "
                "Empty = all available.",
            },
            "batch_size": {
                "type": "integer",
                "description": "Batch size (default: 64)",
            },
            "stride": {
                "type": "integer",
                "description": "Window stride (default: 12)",
            },
        },
        "required": ["timeframe"],
    },
)
def create_data_loader(
    timeframe: str = "5m",
    engineer: str = "",
    assets: str = "",
    batch_size: int = DATA_BATCH_SIZE,
    stride: int = DATA_STRIDE,
) -> str:
    """Create or reconfigure a MarketDataLoader for the given timeframe."""
    try:
        asset_list = json.loads(assets) if assets else None
        loader = get_loader(
            timeframe=timeframe,
            assets=asset_list,
            engineer=engineer or None,
            batch_size=batch_size,
            stride=stride,
            force_new=True,
        )

        tf_cfg = TIMEFRAME_CONFIGS[timeframe]
        asset_names = [a.name if hasattr(a, "name") else str(a) for a in loader.assets_data]
        return json.dumps({
            "status": "ok",
            "timeframe": timeframe,
            "pred_len": tf_cfg["pred_len"],
            "input_len": tf_cfg["input_len"],
            "batch_size": batch_size,
            "stride": stride,
            "engineer": engineer or DATA_FEATURE_ENGINEER,
            "assets_loaded": asset_names,
            "repo": HF_TRAINING_DATA_REPO,
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool(
    description=(
        "Show available timeframes, assets, and current data loader status. "
        "Returns HF repo info, supported timeframes with their prediction horizons, "
        "and the SN50-to-HF asset name mapping."
    ),
)
def data_loader_info() -> str:
    """Return data loader configuration and status."""
    active = {}
    for key, loader in _loaders.items():
        try:
            asset_names = [
                a.name if hasattr(a, "name") else str(a) for a in loader.assets_data
            ]
        except Exception:
            asset_names = ["(error reading assets)"]
        active[key] = {"assets": asset_names}

    return json.dumps({
        "hf_repo": HF_TRAINING_DATA_REPO,
        "timeframes": {
            tf: {"pred_len": cfg["pred_len"], "input_len": cfg["input_len"]}
            for tf, cfg in TIMEFRAME_CONFIGS.items()
        },
        "asset_mapping": SN50_TO_HF_ASSET,
        "available_engineers": ["zscore", "wavelet"],
        "defaults": {
            "batch_size": DATA_BATCH_SIZE,
            "stride": DATA_STRIDE,
            "gap_handling": DATA_GAP_HANDLING,
            "engineer": DATA_FEATURE_ENGINEER,
        },
        "active_loaders": active,
    }, indent=2)


@tool(
    description=(
        "Get train/val/test DataLoaders from the active MarketDataLoader using "
        "static holdout split. timeframe: '5m' or '1m'. "
        "cutoff: fraction of data for train (e.g. 0.7). "
        "val_size: fraction of remaining for validation (e.g. 0.15). "
        "Returns split sizes and sample batch shapes."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "timeframe": {
                "type": "string",
                "enum": ["5m", "1m"],
                "description": "Which loader to split",
            },
            "cutoff": {
                "type": "number",
                "description": "Train fraction (default: 0.7)",
            },
            "val_size": {
                "type": "number",
                "description": "Val fraction of post-cutoff data (default: 0.5)",
            },
        },
        "required": ["timeframe"],
    },
)
def split_data(
    timeframe: str = "5m",
    cutoff: float = 0.7,
    val_size: float = 0.5,
) -> str:
    """Create train/val/test split from the active loader."""
    try:
        # Reuse cached loader or auto-create with defaults
        loader = get_loader(timeframe=timeframe)

        train_dl, val_dl, test_dl = loader.static_holdout(
            cutoff=cutoff, val_size=val_size
        )

        def _dl_info(dl: Any) -> dict:
            try:
                return {"num_batches": len(dl), "batch_size": dl.batch_size}
            except Exception:
                return {"num_batches": "unknown"}

        return json.dumps({
            "status": "ok",
            "timeframe": timeframe,
            "train": _dl_info(train_dl),
            "val": _dl_info(val_dl),
            "test": _dl_info(test_dl),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
