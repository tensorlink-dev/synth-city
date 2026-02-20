"""
Live validator runner — scores baseline models (GBM, Heston) plus research
models from the pipeline, uploading any model that tops the leaderboard.

Flow:
    1. Fetch live prices from Pyth → fit baseline models on recent history
    2. Load any trained research models from the pipeline
    3. Generate 1000 Monte Carlo paths per model per asset
    4. Record price snapshots every ``price_poll_interval`` seconds
    5. After the prediction window elapses, score all models via CRPS
    6. If a research model beats the current leader → publish via research API
    7. Update the leaderboard and dashboard
    8. Repeat
"""

from __future__ import annotations

import json
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import SN50_ASSETS, SN50_NUM_PATHS, WORKSPACE_DIR
from data.market import get_close_prices
from live_validator.leaderboard import Leaderboard
from live_validator.price_feed import PriceFeed
from live_validator.scoring import (
    HIGH_FREQUENCY,
    LOW_FREQUENCY,
    ScoringConfig,
    score_models,
)
from models.base import BaseForecaster
from models.gbm import GBMForecaster
from models.stochastic_vol import HestonForecaster

logger = logging.getLogger(__name__)

# Baseline model tag
BASELINE_TAG = "baseline"
RESEARCH_TAG = "research"


# ---------------------------------------------------------------------------
# Baseline models — fixed reference points
# ---------------------------------------------------------------------------

def build_baselines() -> dict[str, BaseForecaster]:
    """Instantiate the baseline models (GBM + Heston)."""
    return {
        "GBM (baseline)": GBMForecaster(),
        "Heston-SV (baseline)": HestonForecaster(),
    }


# ---------------------------------------------------------------------------
# Research model adapter — wraps open-synth-miner experiments
# ---------------------------------------------------------------------------

class ResearchModelAdapter(BaseForecaster):
    """Wraps a trained open-synth-miner model as a BaseForecaster.

    The model object must be injected via ``set_model()`` after training
    through the ResearchSession — we cannot recreate trained weights
    from config alone, as create_model() only builds the architecture.
    """

    def __init__(
        self,
        name: str,
        experiment_config: dict,
        crps_score: float | None = None,
    ) -> None:
        self.name = name
        self.experiment_config = experiment_config
        self.crps_score = crps_score
        self._model = None

    def set_model(self, model: object) -> None:
        """Inject a trained model object from the ResearchSession."""
        self._model = model

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def fit(self, prices: np.ndarray, **kwargs) -> None:
        """No-op — research models are already trained."""

    def generate_paths(
        self,
        asset: str,
        num_paths: int,
        num_steps: int,
        s0: float | None = None,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(
                f"Research model {self.name} has no trained weights. "
                "Models must be trained via ResearchSession.run() first."
            )
        return self._model.generate_paths(
            asset=asset,
            num_paths=num_paths,
            num_steps=num_steps,
            s0=s0,
        )

    def params_dict(self) -> dict:
        return {
            "model": self.name,
            "type": "research",
            "crps_score": self.crps_score,
            "config": self.experiment_config,
        }


def load_research_models() -> dict[str, ResearchModelAdapter]:
    """Load top experiment configs and train them through the session.

    The ResearchSession is an in-memory singleton. We pull the best
    experiment configs from Hippius (persisted from prior pipeline runs),
    then re-train them through the session to get actual model objects.

    If the session already has results (same-process as pipeline), we
    use those directly instead of re-training.

    Returns only models that trained successfully.
    """
    models: dict[str, ResearchModelAdapter] = {}

    # Strategy 1: check if the session already has trained results
    try:
        from pipeline.tools.research_tools import _get_session
        session = _get_session()
        comparison = session.compare()
        ranking = comparison.get("ranking", [])

        if ranking:
            logger.info(
                "Found %d experiments in active session", len(ranking)
            )
            for entry in ranking[:5]:
                name = entry.get("name", "research-model")
                config = entry.get("experiment") or entry.get("config")
                crps = entry.get("crps")
                # The session may expose trained models — try to get them
                trained_model = entry.get("model")
                if config and isinstance(config, dict):
                    adapter = ResearchModelAdapter(
                        name=name,
                        experiment_config=config,
                        crps_score=crps,
                    )
                    if trained_model is not None:
                        adapter.set_model(trained_model)
                        models[name] = adapter
                        logger.info(
                            "Loaded trained model: %s (CRPS=%.4f)",
                            name, crps or 0.0,
                        )
                    else:
                        logger.debug(
                            "Session has config for %s but no trained "
                            "model object — skipping (would need "
                            "re-training)", name,
                        )
    except Exception as exc:
        logger.debug("Session check: %s", exc)

    if models:
        return models

    # Strategy 2: load configs from Hippius and re-train through session
    try:
        from pipeline.tools.hippius_store import load_hippius_history
        from pipeline.tools.research_tools import _get_session

        history_json = load_hippius_history(limit=5)
        history = json.loads(history_json)
        experiments = history.get("experiments", [])

        if not experiments:
            logger.debug("No experiments in Hippius history")
            return models

        session = _get_session()
        for exp_info in experiments[:3]:  # Re-train top 3
            config = exp_info.get("experiment") or exp_info.get("config")
            name = exp_info.get("name", "hippius-model")
            if not config or not isinstance(config, dict):
                continue

            logger.info("Re-training %s from Hippius config...", name)
            try:
                result = session.run(config, epochs=5, name=name)
                if isinstance(result, dict) and result.get("status") == "ok":
                    crps = result.get("metrics", {}).get("crps")
                    trained_model = result.get("model")
                    adapter = ResearchModelAdapter(
                        name=name,
                        experiment_config=config,
                        crps_score=crps,
                    )
                    if trained_model is not None:
                        adapter.set_model(trained_model)
                        models[name] = adapter
                        logger.info(
                            "Trained %s: CRPS=%.4f", name, crps or 0.0,
                        )
                    else:
                        logger.warning(
                            "Training %s succeeded but session did not "
                            "return model object — cannot use for "
                            "live predictions", name,
                        )
            except Exception as exc:
                logger.warning("Re-training %s failed: %s", name, exc)
    except Exception as exc:
        logger.debug("Hippius re-training: %s", exc)

    return models


# ---------------------------------------------------------------------------
# Auto-publish logic
# ---------------------------------------------------------------------------

def publish_leader(
    model_name: str,
    crps_score: float,
    experiment_config: dict | None,
) -> bool:
    """Publish a model that topped the leaderboard.

    Returns True if successfully published.
    """
    if experiment_config is None:
        logger.info(
            "Model %s is a baseline — nothing to publish", model_name
        )
        return False

    try:
        from pipeline.tools.publish_tools import publish_model
        result_json = publish_model(
            experiment=json.dumps(experiment_config),
            crps_score=crps_score,
        )
        result = json.loads(result_json)
        if result.get("status") == "published":
            logger.info(
                "Published %s (CRPS=%.4f) → %s",
                model_name, crps_score,
                result.get("hf_link", "HF Hub"),
            )
            return True
        logger.warning(
            "Publish returned non-success for %s: %s",
            model_name, result.get("error", "unknown"),
        )
    except Exception as exc:
        logger.error("Failed to publish %s: %s", model_name, exc)
    return False


# ---------------------------------------------------------------------------
# Pending prediction + state
# ---------------------------------------------------------------------------

@dataclass
class PendingPrediction:
    """A prediction waiting to be scored once the window elapses."""
    model_name: str
    asset: str
    paths: np.ndarray           # (num_sims, num_steps)
    start_time: float           # Unix timestamp
    scoring_config: ScoringConfig
    num_steps: int

    @property
    def end_time(self) -> float:
        return self.start_time + self.scoring_config.time_length

    @property
    def is_ready_to_score(self) -> bool:
        return time.time() >= self.end_time


@dataclass
class ValidatorState:
    """Runtime state for the live validator."""
    baselines: dict[str, BaseForecaster] = field(default_factory=dict)
    research_models: dict[str, ResearchModelAdapter] = field(
        default_factory=dict
    )
    pending: list[PendingPrediction] = field(default_factory=list)
    prices_at_start: dict[str, float] = field(default_factory=dict)
    last_prediction_time: float = 0.0
    last_fit_time: float = 0.0
    last_research_reload: float = 0.0
    rounds_completed: int = 0
    total_scores: int = 0
    running: bool = True
    published_models: set[str] = field(default_factory=set)

    @property
    def all_models(self) -> dict[str, BaseForecaster]:
        """Merged view of baselines + ready research models."""
        merged: dict[str, BaseForecaster] = {}
        merged.update(self.baselines)
        for name, model in self.research_models.items():
            if model.is_ready:
                merged[name] = model
        return merged


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

class LiveValidator:
    """Live validator that scores baselines and research models.

    Baselines (GBM, Heston) are fixed references. Research models are
    loaded from the pipeline's ResearchSession and compete against them.
    When a research model tops the board, it's auto-published.

    Parameters
    ----------
    assets : list of str
        Assets to predict. Defaults to all SN50 assets.
    mode : str
        "low" for 24h predictions or "high" for 1h.
    prediction_interval : int
        Seconds between prediction rounds.
    price_poll_interval : int
        Seconds between price recordings.
    refit_interval : int
        Seconds between baseline refits.
    research_reload_interval : int
        Seconds between research model reloads.
    auto_publish : bool
        Publish models that beat all baselines.
    save_dir : Path or None
        Directory for leaderboard + dashboard state.
    """

    def __init__(
        self,
        assets: list[str] | None = None,
        mode: str = "low",
        prediction_interval: int = 3600,
        price_poll_interval: int = 60,
        refit_interval: int = 3600,
        research_reload_interval: int = 1800,
        auto_publish: bool = True,
        save_dir: Path | None = None,
    ) -> None:
        self.assets = assets or list(SN50_ASSETS.keys())
        self.scoring_config = (
            LOW_FREQUENCY if mode == "low" else HIGH_FREQUENCY
        )
        self.prediction_interval = prediction_interval
        self.price_poll_interval = price_poll_interval
        self.refit_interval = refit_interval
        self.research_reload_interval = research_reload_interval
        self.auto_publish = auto_publish
        self.save_dir = save_dir or (WORKSPACE_DIR / "live_validator")

        self.price_feed = PriceFeed(
            assets=self.assets, max_history=2000,
        )
        self.state = ValidatorState(baselines=build_baselines())
        self.leaderboard = Leaderboard(
            save_path=self.save_dir / "leaderboard.json",
        )

        # Register baselines
        for name in self.state.baselines:
            self.leaderboard.register_model(name, is_baseline=True)

        # Graceful shutdown
        self._shutdown = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame: object) -> None:
        logger.info(
            "Received signal %d — shutting down gracefully...", signum
        )
        self._shutdown = True
        self.state.running = False

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def fit_baselines(self) -> None:
        """Fit baseline models on recent historical price data."""
        logger.info("Fitting baselines on historical data...")
        for asset in self.assets:
            try:
                prices = get_close_prices(asset, days=30)
                if len(prices) < 20:
                    logger.warning(
                        "Not enough data for %s (%d points), skipping",
                        asset, len(prices),
                    )
                    continue
                for name, model in self.state.baselines.items():
                    try:
                        model.fit(prices)
                        logger.info(
                            "  Fitted %s on %s (%d pts)",
                            name, asset, len(prices),
                        )
                    except Exception as exc:
                        logger.error(
                            "  Failed to fit %s on %s: %s",
                            name, asset, exc,
                        )
            except Exception as exc:
                logger.warning(
                    "Could not fetch data for %s: %s", asset, exc
                )
        self.state.last_fit_time = time.time()

    def refit_baselines_from_feed(self) -> None:
        """Refit baselines using accumulated price feed data."""
        for asset in self.assets:
            hist = self.price_feed.get_historical_prices(
                asset, count=200
            )
            if hist is None:
                continue
            for name, model in self.state.baselines.items():
                try:
                    model.fit(hist)
                except Exception as exc:
                    logger.error(
                        "Refit failed for %s on %s: %s",
                        name, asset, exc,
                    )
        self.state.last_fit_time = time.time()

    # ------------------------------------------------------------------
    # Research model management
    # ------------------------------------------------------------------

    def reload_research_models(self) -> None:
        """Reload research models from the pipeline session.

        Merges new models into the existing set rather than replacing,
        so models with pending predictions aren't orphaned.
        """
        new_models = load_research_models()
        if new_models:
            self.state.research_models.update(new_models)
            for name in new_models:
                self.leaderboard.register_model(name, is_baseline=False)
            logger.info(
                "Loaded %d research models: %s",
                len(new_models), ", ".join(new_models.keys()),
            )
        self.state.last_research_reload = time.time()

    # ------------------------------------------------------------------
    # Prediction generation
    # ------------------------------------------------------------------

    def generate_predictions(self) -> int:
        """Generate predictions from all models for all assets."""
        num_steps = (
            self.scoring_config.time_length
            // self.scoring_config.time_increment + 1
        )
        start_time = time.time()
        count = 0

        current_prices = self.price_feed.fetch_all_prices()
        self.state.prices_at_start = current_prices

        all_models = self.state.all_models
        for asset in self.assets:
            s0 = current_prices.get(asset)
            if s0 is None:
                logger.warning(
                    "No live price for %s — skipping", asset
                )
                continue

            for model_name, model in all_models.items():
                try:
                    paths = model.generate_paths(
                        asset=asset,
                        num_paths=SN50_NUM_PATHS,
                        num_steps=num_steps,
                        s0=s0,
                    )
                    if paths.shape != (SN50_NUM_PATHS, num_steps):
                        logger.warning(
                            "Shape mismatch for %s/%s: %s",
                            model_name, asset, paths.shape,
                        )
                        continue

                    self.state.pending.append(PendingPrediction(
                        model_name=model_name,
                        asset=asset,
                        paths=paths,
                        start_time=start_time,
                        scoring_config=self.scoring_config,
                        num_steps=num_steps,
                    ))
                    count += 1
                except Exception as exc:
                    logger.error(
                        "Prediction failed for %s/%s: %s",
                        model_name, asset, exc,
                    )

        self.state.last_prediction_time = start_time
        logger.info(
            "Generated %d predictions (%d models x %d assets) at %s",
            count, len(all_models), len(self.assets),
            datetime.fromtimestamp(
                start_time, tz=timezone.utc
            ).strftime("%H:%M:%S UTC"),
        )
        return count

    # ------------------------------------------------------------------
    # Scoring + auto-publish
    # ------------------------------------------------------------------

    def score_ready_predictions(self) -> int:
        """Score any predictions whose windows have elapsed."""
        ready = [p for p in self.state.pending if p.is_ready_to_score]
        if not ready:
            return 0

        # Group by (asset, start_time)
        groups: dict[tuple[str, float], list[PendingPrediction]] = {}
        for pred in ready:
            key = (pred.asset, pred.start_time)
            if key not in groups:
                groups[key] = []
            groups[key].append(pred)

        scored_count = 0
        for (asset, start_time), preds in groups.items():
            config = preds[0].scoring_config
            num_steps = preds[0].num_steps

            real_path = self.price_feed.get_price_path(
                asset=asset,
                start_time=start_time,
                time_increment=config.time_increment,
                num_steps=num_steps,
            )

            if real_path is None:
                logger.warning(
                    "Cannot score %s from %.0f — insufficient history",
                    asset, start_time,
                )
                for pred in preds:
                    self.state.pending.remove(pred)
                continue

            model_preds = {p.model_name: p.paths for p in preds}
            results = score_models(model_preds, real_path, config)

            # Log results
            ts_str = datetime.fromtimestamp(
                start_time, tz=timezone.utc
            ).strftime("%H:%M:%S")
            logger.info("Scored %s (started %s):", asset, ts_str)
            for name, data in sorted(
                results.items(), key=lambda x: x[1]["raw_crps"]
            ):
                crps = data["raw_crps"]
                tag = (
                    "baseline" if name in self.state.baselines
                    else "research"
                )
                crps_str = f"{crps:.4f}" if crps >= 0 else "FAILED"
                logger.info(
                    "  %-25s [%s]  CRPS=%s",
                    name, tag, crps_str,
                )

            # Update leaderboard
            self.leaderboard.add_scores(results, asset, config)

            # Remove scored predictions
            for pred in preds:
                self.state.pending.remove(pred)
            scored_count += len(preds)

        self.state.total_scores += scored_count
        self.state.rounds_completed += 1

        # Check if a research model is now the leader → auto-publish
        if self.auto_publish:
            self._maybe_publish_leader()

        return scored_count

    def _maybe_publish_leader(self) -> None:
        """Publish the leader if it's a research model."""
        leader = self.leaderboard.get_leader()
        if leader is None:
            return
        if leader in self.state.baselines:
            return  # Baselines don't need publishing
        if leader in self.state.published_models:
            return  # Already published

        # Get the research model's config
        research_model = self.state.research_models.get(leader)
        if research_model is None:
            return

        stats = self.leaderboard.stats.get(leader)
        if stats is None or stats.total_scores < 2:
            return  # Need at least 2 scores before publishing

        crps = stats.rolling_avg_crps
        logger.info(
            "Research model %s is leading (CRPS=%.4f) — publishing!",
            leader, crps,
        )
        ok = publish_leader(
            leader, crps, research_model.experiment_config
        )
        if ok:
            self.state.published_models.add(leader)

    # ------------------------------------------------------------------
    # Status display
    # ------------------------------------------------------------------

    def print_status(self) -> None:
        """Print current validator status and leaderboard."""
        now = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        pending_count = len(self.state.pending)
        oldest_pending = ""
        if self.state.pending:
            remaining = max(
                0, self.state.pending[0].end_time - time.time()
            )
            oldest_pending = f" (next score in {remaining:.0f}s)"

        n_base = len(self.state.baselines)
        n_research = len(self.state.research_models)
        logger.info(
            "\n[%s] Round %d | %d scored | %d pending%s | "
            "%d baselines + %d research",
            now, self.state.rounds_completed, self.state.total_scores,
            pending_count, oldest_pending, n_base, n_research,
        )
        logger.info(self.leaderboard.format_table())

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the live validator continuously."""
        logger.info("=" * 70)
        logger.info("LIVE VALIDATOR — baselines vs research models")
        logger.info("Mode: %s frequency", self.scoring_config.label)
        logger.info("Assets: %s", ", ".join(self.assets))
        logger.info(
            "Baselines: %s", ", ".join(self.state.baselines.keys())
        )
        logger.info("Auto-publish: %s", self.auto_publish)
        logger.info("Prediction interval: %ds", self.prediction_interval)
        logger.info("Price poll interval: %ds", self.price_poll_interval)
        window = self.scoring_config.time_length
        logger.info(
            "Prediction window: %ds (%s)", window,
            f"{window // 3600}h" if window >= 3600
            else f"{window // 60}m",
        )
        logger.info("=" * 70)

        # Step 1: Fit baselines
        self.fit_baselines()

        # Step 2: Load research models
        self.reload_research_models()

        # Step 3: Seed price feed
        logger.info("Recording initial prices...")
        self.price_feed.record_prices()

        # Step 4: First predictions
        self.generate_predictions()

        last_poll = time.time()
        last_status = time.time()
        status_interval = 120

        logger.info(
            "Recording prices every %ds. "
            "Scores after %ds window elapses.",
            self.price_poll_interval,
            self.scoring_config.time_length,
        )

        while self.state.running and not self._shutdown:
            now = time.time()

            # Record prices
            if now - last_poll >= self.price_poll_interval:
                prices = self.price_feed.record_prices()
                last_poll = now
                if prices:
                    logger.debug(
                        "Recorded prices for %d assets", len(prices)
                    )

            # Score ready predictions
            scored = self.score_ready_predictions()
            if scored > 0:
                self.print_status()
                self.leaderboard.save()

            # New prediction round
            if now - self.state.last_prediction_time >= self.prediction_interval:
                if now - self.state.last_fit_time >= self.refit_interval:
                    self.refit_baselines_from_feed()

                # Reload research models periodically
                if now - self.state.last_research_reload >= self.research_reload_interval:
                    self.reload_research_models()

                self.generate_predictions()

            # Periodic status
            if now - last_status >= status_interval:
                self.print_status()
                last_status = now

            time.sleep(min(5, self.price_poll_interval / 2))

        # Shutdown
        logger.info("Live validator stopped.")
        self.print_status()
        self.leaderboard.save()
        self.price_feed.close()


def run_live_validator(
    assets: list[str] | None = None,
    mode: str = "low",
    prediction_interval: int = 3600,
    price_poll_interval: int = 60,
    refit_interval: int = 3600,
    auto_publish: bool = True,
    dashboard_port: int = 8378,
) -> None:
    """Entry point for the live validator."""
    from live_validator.dashboard import start_dashboard

    validator = LiveValidator(
        assets=assets,
        mode=mode,
        prediction_interval=prediction_interval,
        price_poll_interval=price_poll_interval,
        refit_interval=refit_interval,
        auto_publish=auto_publish,
    )

    # Start dashboard in background
    start_dashboard(
        port=dashboard_port,
        leaderboard_path=validator.save_dir / "leaderboard.json",
    )

    validator.run()
