"""Tests for experiment novelty checking in pipeline.tools.analysis_tools."""

from __future__ import annotations

import json
from typing import Any

from pipeline.tools.analysis_tools import (
    _categorize_error,
    _extract_config_features,
    _find_similar,
    _fingerprint,
    check_experiment_novelty,
    invalidate_novelty_cache,
)

# ---------------------------------------------------------------------------
# Helpers — build experiment configs matching the real format
# ---------------------------------------------------------------------------

def _make_config(
    blocks: list[str] | None = None,
    head: str = "GBMHead",
    d_model: int = 32,
    lr: float = 0.001,
    horizon: int = 288,
    seq_len: int = 288,
    feature_dim: int = 4,
) -> dict[str, Any]:
    """Build a minimal experiment config dict."""
    return {
        "model": {
            "backbone": {
                "blocks": blocks or ["RevIN", "TransformerBlock"],
                "d_model": d_model,
                "feature_dim": feature_dim,
                "seq_len": seq_len,
            },
            "head": {
                "_target_": f"osa.models.heads.{head}",
                "latent_size": d_model,
            },
        },
        "training": {
            "lr": lr,
            "horizon": horizon,
        },
    }


def _make_cache_entry(
    name: str = "exp-1",
    blocks: list[str] | None = None,
    head: str = "GBMHead",
    d_model: int = 32,
    lr: float = 0.001,
    crps: float | None = 0.05,
    status: str = "ok",
    error: str = "",
) -> dict[str, Any]:
    """Build a cache entry as stored by _refresh_fp_cache."""
    return {
        "name": name,
        "run_id": "run-001",
        "timestamp": "2025-02-20T10:00:00",
        "crps": crps,
        "status": status,
        "blocks": blocks or ["RevIN", "TransformerBlock"],
        "head": head,
        "d_model": d_model,
        "lr": lr,
        "error": error,
    }


# ---------------------------------------------------------------------------
# _fingerprint
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_deterministic(self) -> None:
        """Same config always produces the same fingerprint."""
        cfg = _make_config()
        assert _fingerprint(cfg) == _fingerprint(cfg)

    def test_different_blocks_different_fp(self) -> None:
        cfg_a = _make_config(blocks=["LSTMBlock"])
        cfg_b = _make_config(blocks=["GRUBlock"])
        assert _fingerprint(cfg_a) != _fingerprint(cfg_b)

    def test_different_head_different_fp(self) -> None:
        cfg_a = _make_config(head="GBMHead")
        cfg_b = _make_config(head="SDEHead")
        assert _fingerprint(cfg_a) != _fingerprint(cfg_b)

    def test_different_d_model_different_fp(self) -> None:
        cfg_a = _make_config(d_model=32)
        cfg_b = _make_config(d_model=64)
        assert _fingerprint(cfg_a) != _fingerprint(cfg_b)

    def test_different_lr_different_fp(self) -> None:
        cfg_a = _make_config(lr=0.001)
        cfg_b = _make_config(lr=0.0003)
        assert _fingerprint(cfg_a) != _fingerprint(cfg_b)

    def test_hex_length(self) -> None:
        fp = _fingerprint(_make_config())
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)

    def test_empty_config(self) -> None:
        """Empty config should still produce a valid fingerprint."""
        fp = _fingerprint({})
        assert len(fp) == 16


# ---------------------------------------------------------------------------
# _extract_config_features
# ---------------------------------------------------------------------------

class TestExtractConfigFeatures:
    def test_extracts_all_fields(self) -> None:
        cfg = _make_config(
            blocks=["LSTMBlock"], head="SDEHead", d_model=64, lr=0.0003,
            horizon=288, seq_len=96,
        )
        features = _extract_config_features(cfg)
        assert features["blocks"] == ["LSTMBlock"]
        assert features["head"] == "SDEHead"
        assert features["d_model"] == 64
        assert features["lr"] == 0.0003
        assert features["horizon"] == 288
        assert features["seq_len"] == 96

    def test_strips_module_path(self) -> None:
        """Head names like 'osa.models.heads.GBMHead' should be stripped to 'GBMHead'."""
        cfg = _make_config(head="GBMHead")
        features = _extract_config_features(cfg)
        assert features["head"] == "GBMHead"

    def test_empty_config(self) -> None:
        features = _extract_config_features({})
        assert features["blocks"] == []
        assert features["d_model"] is None


# ---------------------------------------------------------------------------
# _categorize_error
# ---------------------------------------------------------------------------

class TestCategorizeError:
    def test_empty_string(self) -> None:
        assert _categorize_error("") == ""

    def test_tensor_shape_mismatch(self) -> None:
        err = "The size of tensor a (12) must match the size of tensor b (288)"
        assert _categorize_error(err) == "tensor shape mismatch"

    def test_d_model_nhead(self) -> None:
        assert _categorize_error("d_model must be divisible by nhead") == (
            "d_model not divisible by nhead"
        )

    def test_oom(self) -> None:
        assert _categorize_error("CUDA out of memory") == "out of memory"

    def test_nan(self) -> None:
        assert _categorize_error("NaN detected in loss") == "NaN loss / degenerate output"

    def test_unknown_component(self) -> None:
        assert _categorize_error("Unknown block: FooBlock") == "unknown component name"

    def test_truncates_long_errors(self) -> None:
        long_err = "x" * 200
        result = _categorize_error(long_err)
        assert len(result) <= 80


# ---------------------------------------------------------------------------
# _find_similar
# ---------------------------------------------------------------------------

class TestFindSimilar:
    def test_no_similar(self) -> None:
        """No matches when cache is empty."""
        target = _extract_config_features(
            _make_config(blocks=["LSTMBlock"], head="SDEHead")
        )
        result = _find_similar(target, {}, "target_fp")
        assert result["similar"] == []
        assert result["warnings"] == []
        assert result["total_similar"] == 0

    def test_same_architecture_different_params(self) -> None:
        """Same blocks + head but different d_model should match as same_architecture."""
        target = _extract_config_features(
            _make_config(blocks=["LSTMBlock"], head="SDEHead", d_model=64)
        )
        cache = {
            "other_fp": [
                _make_cache_entry(
                    blocks=["LSTMBlock"], head="SDEHead", d_model=32, crps=0.04,
                ),
            ],
        }
        result = _find_similar(target, cache, "target_fp")
        assert len(result["similar"]) == 1
        assert result["similar"][0]["match_type"] == "same_architecture"
        assert result["similar"][0]["crps"] == 0.04

    def test_same_blocks_different_head(self) -> None:
        target = _extract_config_features(
            _make_config(blocks=["LSTMBlock"], head="SDEHead")
        )
        cache = {
            "other_fp": [
                _make_cache_entry(blocks=["LSTMBlock"], head="GBMHead", crps=0.03),
            ],
        }
        result = _find_similar(target, cache, "target_fp")
        assert len(result["similar"]) == 1
        assert result["similar"][0]["match_type"] == "same_blocks"

    def test_same_head_different_blocks(self) -> None:
        target = _extract_config_features(
            _make_config(blocks=["LSTMBlock"], head="SDEHead")
        )
        cache = {
            "other_fp": [
                _make_cache_entry(blocks=["GRUBlock"], head="SDEHead", crps=0.06),
            ],
        }
        result = _find_similar(target, cache, "target_fp")
        assert len(result["similar"]) == 1
        assert result["similar"][0]["match_type"] == "same_head"

    def test_no_overlap_excluded(self) -> None:
        """Experiments with completely different blocks and head are excluded."""
        target = _extract_config_features(
            _make_config(blocks=["LSTMBlock"], head="SDEHead")
        )
        cache = {
            "other_fp": [
                _make_cache_entry(blocks=["GRUBlock"], head="GBMHead"),
            ],
        }
        result = _find_similar(target, cache, "target_fp")
        assert result["similar"] == []

    def test_exact_fingerprint_excluded(self) -> None:
        """Exact fingerprint matches are excluded (handled separately)."""
        target = _extract_config_features(
            _make_config(blocks=["LSTMBlock"], head="SDEHead")
        )
        cache = {
            "target_fp": [
                _make_cache_entry(blocks=["LSTMBlock"], head="SDEHead"),
            ],
        }
        result = _find_similar(target, cache, "target_fp")
        assert result["similar"] == []

    def test_failure_warnings(self) -> None:
        """Failed similar experiments produce warnings."""
        target = _extract_config_features(
            _make_config(blocks=["LSTMBlock"], head="SDEHead")
        )
        cache = {
            "other_fp": [
                _make_cache_entry(
                    blocks=["LSTMBlock"], head="NeuralBridgeHead",
                    status="error",
                    error="The size of tensor a (12) must match the size of tensor b (288)",
                    crps=None,
                ),
            ],
        }
        result = _find_similar(target, cache, "target_fp")
        assert len(result["warnings"]) == 1
        assert "tensor shape mismatch" in result["warnings"][0]

    def test_sort_order(self) -> None:
        """same_architecture should come before same_blocks, then same_head."""
        target = _extract_config_features(
            _make_config(blocks=["LSTMBlock"], head="SDEHead", d_model=64)
        )
        cache = {
            "fp_a": [
                _make_cache_entry(
                    blocks=["GRUBlock"], head="SDEHead", crps=0.01,
                ),
            ],
            "fp_b": [
                _make_cache_entry(
                    blocks=["LSTMBlock"], head="SDEHead", d_model=32, crps=0.02,
                ),
            ],
            "fp_c": [
                _make_cache_entry(
                    blocks=["LSTMBlock"], head="GBMHead", crps=0.03,
                ),
            ],
        }
        result = _find_similar(target, cache, "target_fp")
        match_types = [s["match_type"] for s in result["similar"]]
        assert match_types == ["same_architecture", "same_blocks", "same_head"]

    def test_limits_to_five(self) -> None:
        """Result is capped at 5 similar experiments."""
        target = _extract_config_features(
            _make_config(blocks=["LSTMBlock"], head="SDEHead")
        )
        cache = {}
        for i in range(10):
            cache[f"fp_{i}"] = [
                _make_cache_entry(
                    name=f"exp-{i}",
                    blocks=["LSTMBlock"], head="SDEHead",
                    d_model=32 + i, crps=0.05 + i * 0.01,
                ),
            ]
        result = _find_similar(target, cache, "target_fp")
        assert len(result["similar"]) == 5
        assert result["total_similar"] == 10

    def test_warnings_limited_to_three(self) -> None:
        """Warnings are capped at 3."""
        target = _extract_config_features(
            _make_config(blocks=["LSTMBlock"], head="SDEHead")
        )
        cache = {}
        for i in range(5):
            cache[f"fp_{i}"] = [
                _make_cache_entry(
                    name=f"exp-{i}",
                    blocks=["LSTMBlock"], head="SDEHead",
                    d_model=32 + i,
                    status="error",
                    error=f"Error variant {i}",
                    crps=None,
                ),
            ]
        result = _find_similar(target, cache, "target_fp")
        assert len(result["warnings"]) == 3


# ---------------------------------------------------------------------------
# invalidate_novelty_cache
# ---------------------------------------------------------------------------

class TestInvalidateNoveltyCache:
    def test_resets_timestamp(self) -> None:
        import pipeline.tools.analysis_tools as mod
        mod._fp_cache_ts = 9999.0
        invalidate_novelty_cache()
        assert mod._fp_cache_ts == 0.0

    def test_preserves_cache_data(self) -> None:
        """Invalidation should only reset the TTL, not clear existing data."""
        import pipeline.tools.analysis_tools as mod
        mod._fp_cache = {"some_fp": [{"name": "exp"}]}
        mod._fp_cache_ts = 9999.0
        invalidate_novelty_cache()
        assert mod._fp_cache == {"some_fp": [{"name": "exp"}]}
        # Clean up
        mod._fp_cache = {}


# ---------------------------------------------------------------------------
# check_experiment_novelty (integration-style, mocked Hippius)
# ---------------------------------------------------------------------------

class TestCheckExperimentNovelty:
    def _run_check(
        self,
        config: dict[str, Any],
        cache: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Run check_experiment_novelty with a mocked cache."""
        import pipeline.tools.analysis_tools as mod
        import pipeline.tools.hippius_store as hs_mod
        old_cache = mod._fp_cache
        old_ts = mod._fp_cache_ts
        old_unreachable = hs_mod._endpoint_unreachable
        try:
            mod._fp_cache = cache
            mod._fp_cache_ts = 1e12  # far future — prevents refresh
            hs_mod._endpoint_unreachable = False
            result_str = check_experiment_novelty(json.dumps(config))
            return json.loads(result_str)
        finally:
            mod._fp_cache = old_cache
            mod._fp_cache_ts = old_ts
            hs_mod._endpoint_unreachable = old_unreachable

    def test_novel_experiment_empty_cache(self) -> None:
        config = _make_config(blocks=["LSTMBlock"], head="SDEHead")
        result = self._run_check(config, {})
        assert result["is_novel"] is True
        assert result["recommendation"] == "Novel config — proceed with training."

    def test_exact_duplicate(self) -> None:
        config = _make_config(blocks=["LSTMBlock"], head="SDEHead")
        fp = _fingerprint(config)
        cache = {
            fp: [
                _make_cache_entry(
                    blocks=["LSTMBlock"], head="SDEHead", crps=0.04,
                ),
            ],
        }
        result = self._run_check(config, cache)
        assert result["is_novel"] is False
        assert result["times_tried"] == 1
        assert result["best_prior_crps"] == 0.04

    def test_novel_with_similar_successful(self) -> None:
        """Novel config with similar successful experiments reports their CRPS."""
        config = _make_config(blocks=["LSTMBlock"], head="SDEHead", d_model=64)
        cache = {
            "other_fp": [
                _make_cache_entry(
                    blocks=["LSTMBlock"], head="SDEHead", d_model=32, crps=0.03,
                ),
            ],
        }
        result = self._run_check(config, cache)
        assert result["is_novel"] is True
        assert "similar_experiments" in result
        assert "CRPS=0.03" in result["recommendation"]

    def test_novel_with_failed_similar_warns(self) -> None:
        """Novel config with failed similar experiments includes warnings."""
        config = _make_config(blocks=["LSTMBlock"], head="SDEHead", d_model=64)
        cache = {
            "other_fp": [
                _make_cache_entry(
                    blocks=["LSTMBlock"], head="NeuralBridgeHead",
                    status="error",
                    error="The size of tensor a (12) must match the size of tensor b (288)",
                    crps=None,
                ),
            ],
        }
        result = self._run_check(config, cache)
        assert result["is_novel"] is True
        assert "warnings" in result
        assert "Review the warnings" in result["recommendation"]

    def test_accepts_dict_input(self) -> None:
        """check_experiment_novelty should handle both str and dict input."""
        import pipeline.tools.analysis_tools as mod
        import pipeline.tools.hippius_store as hs_mod
        old_cache = mod._fp_cache
        old_ts = mod._fp_cache_ts
        old_unreachable = hs_mod._endpoint_unreachable
        try:
            mod._fp_cache = {}
            mod._fp_cache_ts = 1e12
            hs_mod._endpoint_unreachable = False
            # Pass a dict instead of JSON string
            result_str = check_experiment_novelty(_make_config())  # type: ignore[arg-type]
            result = json.loads(result_str)
            assert result["is_novel"] is True
        finally:
            mod._fp_cache = old_cache
            mod._fp_cache_ts = old_ts
            hs_mod._endpoint_unreachable = old_unreachable

    def test_error_handling(self) -> None:
        """Invalid JSON should return an error dict, not raise."""
        result_str = check_experiment_novelty("not valid json {{{")
        result = json.loads(result_str)
        assert "error" in result

    def test_hippius_unreachable_no_cache(self) -> None:
        """When Hippius is unreachable and no cache, proceed as novel."""
        import pipeline.tools.analysis_tools as mod
        import pipeline.tools.hippius_store as hs_mod
        old_cache = mod._fp_cache
        old_ts = mod._fp_cache_ts
        old_unreachable = hs_mod._endpoint_unreachable
        try:
            mod._fp_cache = {}
            mod._fp_cache_ts = 0.0
            hs_mod._endpoint_unreachable = True
            result_str = check_experiment_novelty(
                json.dumps(_make_config())
            )
            result = json.loads(result_str)
            assert result["is_novel"] is True
            assert "Cannot verify" in result.get("note", "")
        finally:
            mod._fp_cache = old_cache
            mod._fp_cache_ts = old_ts
            hs_mod._endpoint_unreachable = old_unreachable
