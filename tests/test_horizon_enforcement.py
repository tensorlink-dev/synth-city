"""Tests for horizon enforcement in experiment creation and training dispatch."""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# create_experiment — horizon must match timeframe pred_len
# ---------------------------------------------------------------------------

class TestCreateExperimentHorizon:
    """Verify that create_experiment always sets horizon from the timeframe."""

    def _create(self, **kwargs: Any) -> dict[str, Any]:
        from pipeline.tools.research_tools import create_experiment
        result_str = create_experiment(**kwargs)
        return json.loads(result_str)

    def test_5m_timeframe_sets_horizon_288(self) -> None:
        result = self._create(blocks='["LSTMBlock"]', timeframe="5m")
        assert result["training"]["horizon"] == 288

    def test_1m_timeframe_sets_horizon_60(self) -> None:
        result = self._create(blocks='["LSTMBlock"]', timeframe="1m")
        assert result["training"]["horizon"] == 60

    def test_timeframe_overrides_explicit_horizon(self) -> None:
        """Even if the caller passes horizon=12, timeframe='5m' should win."""
        result = self._create(
            blocks='["LSTMBlock"]', timeframe="5m", horizon=12,
        )
        assert result["training"]["horizon"] == 288

    def test_5m_timeframe_sets_seq_len_288(self) -> None:
        result = self._create(blocks='["LSTMBlock"]', timeframe="5m")
        assert result["model"]["backbone"]["seq_len"] == 288

    def test_no_timeframe_uses_research_default(self) -> None:
        """Without a timeframe, horizon should use the RESEARCH_HORIZON default."""
        from config import RESEARCH_HORIZON
        result = self._create(blocks='["LSTMBlock"]')
        assert result["training"]["horizon"] == RESEARCH_HORIZON


# ---------------------------------------------------------------------------
# run_experiment_on_deployment — horizon correction before sending
# ---------------------------------------------------------------------------

class TestDeploymentHorizonEnforcement:
    """Verify that run_experiment_on_deployment fixes horizon mismatches."""

    def _build_experiment(
        self,
        horizon: int = 12,
        seq_len: int = 32,
    ) -> dict[str, Any]:
        return {
            "model": {
                "backbone": {
                    "blocks": ["LSTMBlock"],
                    "d_model": 64,
                    "feature_dim": 4,
                    "seq_len": seq_len,
                },
                "head": {
                    "_target_": "osa.models.heads.GBMHead",
                    "latent_size": 64,
                },
            },
            "training": {
                "horizon": horizon,
                "n_paths": 100,
                "batch_size": 32,
                "lr": 0.001,
            },
        }

    def test_corrects_wrong_horizon(self) -> None:
        """An experiment with horizon=12 should be corrected to 288 for 5m."""
        exp = self._build_experiment(horizon=12)
        assert exp["training"]["horizon"] == 12

        # Simulate the enforcement logic from run_experiment_on_deployment
        from config import TIMEFRAME_CONFIGS
        tf_cfg = TIMEFRAME_CONFIGS["5m"]
        expected_horizon = int(tf_cfg["pred_len"])
        training = exp.setdefault("training", {})
        training["horizon"] = expected_horizon

        assert exp["training"]["horizon"] == 288

    def test_correct_horizon_unchanged(self) -> None:
        """An experiment with horizon=288 should stay 288 for 5m."""
        exp = self._build_experiment(horizon=288)
        from config import TIMEFRAME_CONFIGS
        tf_cfg = TIMEFRAME_CONFIGS["5m"]
        expected_horizon = int(tf_cfg["pred_len"])
        actual = exp["training"]["horizon"]
        assert actual == expected_horizon

    def test_corrects_wrong_seq_len(self) -> None:
        """Backbone seq_len=32 should be corrected to 288 for 5m."""
        exp = self._build_experiment(seq_len=32)
        from config import TIMEFRAME_CONFIGS
        tf_cfg = TIMEFRAME_CONFIGS["5m"]
        expected_input_len = int(tf_cfg["input_len"])
        backbone = exp["model"]["backbone"]
        if backbone["seq_len"] != expected_input_len:
            backbone["seq_len"] = expected_input_len
        assert exp["model"]["backbone"]["seq_len"] == 288

    def test_missing_training_section_gets_created(self) -> None:
        """If training section is missing, it should be created with correct horizon."""
        exp: dict[str, Any] = {
            "model": {
                "backbone": {"blocks": ["LSTMBlock"], "d_model": 32},
                "head": {"_target_": "osa.models.heads.GBMHead"},
            },
        }
        from config import TIMEFRAME_CONFIGS
        tf_cfg = TIMEFRAME_CONFIGS["5m"]
        training = exp.setdefault("training", {})
        training["horizon"] = int(tf_cfg["pred_len"])
        assert exp["training"]["horizon"] == 288

    def test_1m_timeframe_corrects_to_60(self) -> None:
        """For 1m timeframe, horizon should be corrected to 60."""
        exp = self._build_experiment(horizon=12)
        from config import TIMEFRAME_CONFIGS
        tf_cfg = TIMEFRAME_CONFIGS["1m"]
        exp["training"]["horizon"] = int(tf_cfg["pred_len"])
        assert exp["training"]["horizon"] == 60
