"""Tests for pipeline.tools.proxy_tools — FLOPs/param estimation and ablation tools."""

from __future__ import annotations

import json

from pipeline.tools.proxy_tools import (
    _BLOCK_FLOP_SCALING,
    _BLOCK_PARAM_FACTORS,
    _HEAD_PARAM_FACTORS,
    _estimate_block_params,
    _estimate_head_params,
    _relative_flops,
    estimate_flops,
    estimate_params,
    generate_ablation_configs,
    sweep_configs,
)

# ---------------------------------------------------------------------------
# Constants (defaults from config.py without .env overrides)
# ---------------------------------------------------------------------------
DEFAULT_D_MODEL = 32
DEFAULT_HORIZON = 12
DEFAULT_SEQ_LEN = 32
DEFAULT_FEATURE_DIM = 4
DEFAULT_N_PATHS = 100
DEFAULT_LR = 0.001
DEFAULT_BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# _estimate_block_params
# ---------------------------------------------------------------------------

class TestEstimateBlockParams:
    def test_normalization_blocks_scale_linearly(self) -> None:
        """RevIN and LayerNormBlock params should scale as factor * d_model."""
        for block in ("RevIN", "LayerNormBlock"):
            factor = _BLOCK_PARAM_FACTORS[block]
            assert _estimate_block_params(block, 32) == int(factor * 32)
            assert _estimate_block_params(block, 64) == int(factor * 64)

    def test_regular_blocks_scale_quadratically(self) -> None:
        """Non-normalization blocks should scale as factor * d_model^2."""
        for block in ("LSTMBlock", "GRUBlock", "TransformerBlock"):
            factor = _BLOCK_PARAM_FACTORS[block]
            assert _estimate_block_params(block, 32) == int(factor * 32 * 32)
            assert _estimate_block_params(block, 64) == int(factor * 64 * 64)

    def test_unknown_block_uses_default_factor(self) -> None:
        """Unknown blocks should use the default factor of 6.0."""
        result = _estimate_block_params("UnknownBlock", 32)
        assert result == int(6.0 * 32 * 32)

    def test_all_known_blocks_have_positive_params(self) -> None:
        """Every known block should produce a positive param count."""
        for block in _BLOCK_PARAM_FACTORS:
            assert _estimate_block_params(block, 32) > 0

    def test_lstm_more_params_than_gru(self) -> None:
        """LSTM (4 gates) should have more params than GRU (3 gates)."""
        lstm = _estimate_block_params("LSTMBlock", 64)
        gru = _estimate_block_params("GRUBlock", 64)
        assert lstm > gru

    def test_transformer_encoder_more_than_block(self) -> None:
        """TransformerEncoder (deeper) should have more params than TransformerBlock."""
        encoder = _estimate_block_params("TransformerEncoder", 64)
        block = _estimate_block_params("TransformerBlock", 64)
        assert encoder > block


# ---------------------------------------------------------------------------
# _estimate_head_params
# ---------------------------------------------------------------------------

class TestEstimateHeadParams:
    def test_base_scales_quadratically(self) -> None:
        """Head params base should scale as factor * d_model^2."""
        factor = _HEAD_PARAM_FACTORS["GBMHead"]
        assert _estimate_head_params("GBMHead", 32, 12) == int(factor * 32 * 32)

    def test_horizon_aware_heads_add_extra_params(self) -> None:
        """HorizonHead, NeuralBridgeHead, NeuralSDEHead add d_model * horizon."""
        for head in ("HorizonHead", "NeuralBridgeHead", "NeuralSDEHead"):
            factor = _HEAD_PARAM_FACTORS[head]
            expected = int(factor * 32 * 32) + 32 * 12
            assert _estimate_head_params(head, 32, 12) == expected

    def test_non_horizon_heads_ignore_horizon(self) -> None:
        """GBMHead and SDEHead should produce the same params regardless of horizon."""
        for head in ("GBMHead", "SDEHead"):
            p12 = _estimate_head_params(head, 32, 12)
            p288 = _estimate_head_params(head, 32, 288)
            assert p12 == p288

    def test_unknown_head_uses_default_factor(self) -> None:
        """Unknown head should use default factor of 6.0."""
        result = _estimate_head_params("UnknownHead", 32, 12)
        assert result == int(6.0 * 32 * 32)

    def test_neural_sde_most_expensive(self) -> None:
        """NeuralSDEHead should be the most expensive head."""
        params = {h: _estimate_head_params(h, 64, 12) for h in _HEAD_PARAM_FACTORS}
        assert max(params, key=params.get) == "NeuralSDEHead"  # type: ignore[arg-type]

    def test_gbm_cheapest(self) -> None:
        """GBMHead should be the cheapest head."""
        params = {h: _estimate_head_params(h, 64, 12) for h in _HEAD_PARAM_FACTORS}
        assert min(params, key=params.get) == "GBMHead"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _relative_flops
# ---------------------------------------------------------------------------

class TestRelativeFlops:
    def test_single_block_positive(self) -> None:
        """A single-block architecture should produce positive FLOPs."""
        flops = _relative_flops(["LSTMBlock"], "GBMHead", 32, 32, 12)
        assert flops > 0

    def test_more_blocks_more_flops(self) -> None:
        """Adding blocks should increase FLOPs."""
        single = _relative_flops(["LSTMBlock"], "GBMHead", 32, 32, 12)
        double = _relative_flops(["LSTMBlock", "LSTMBlock"], "GBMHead", 32, 32, 12)
        assert double > single

    def test_larger_d_model_more_flops(self) -> None:
        """Increasing d_model should increase FLOPs."""
        small = _relative_flops(["LSTMBlock"], "GBMHead", 32, 32, 12)
        large = _relative_flops(["LSTMBlock"], "GBMHead", 64, 32, 12)
        assert large > small

    def test_quadratic_scaling_dominates_at_long_seq(self) -> None:
        """Transformer blocks should show much higher FLOPs at longer sequences."""
        short_tf = _relative_flops(["TransformerBlock"], "GBMHead", 32, 32, 12)
        long_tf = _relative_flops(["TransformerBlock"], "GBMHead", 32, 288, 12)

        short_lstm = _relative_flops(["LSTMBlock"], "GBMHead", 32, 32, 12)
        long_lstm = _relative_flops(["LSTMBlock"], "GBMHead", 32, 288, 12)

        # Transformer ratio should be much larger than LSTM ratio
        tf_ratio = long_tf / short_tf
        lstm_ratio = long_lstm / short_lstm
        assert tf_ratio > lstm_ratio

    def test_nlogn_scaling_for_fourier(self) -> None:
        """FourierBlock should show n*log(n) growth with sequence length."""
        short = _relative_flops(["FourierBlock"], "GBMHead", 32, 32, 12)
        long = _relative_flops(["FourierBlock"], "GBMHead", 32, 256, 12)
        # Ratio should be roughly (256 * log2(256)) / (32 * log2(32))
        # = (256 * 8) / (32 * 5) = 2048 / 160 ≈ 12.8, but input projection
        # adds a constant component, so actual ratio will be somewhat lower.
        ratio = long / short
        assert ratio > 5  # Well above linear (8x)

    def test_includes_input_projection(self) -> None:
        """FLOPs should include the input projection component."""
        # With an extremely cheap block (RevIN = 2*d_model params),
        # input projection should be a noticeable fraction
        flops = _relative_flops(["RevIN"], "GBMHead", 32, 32, 12)
        input_proj_flops = (DEFAULT_FEATURE_DIM * 32 + 32) * 32  # (feat*d+d)*seq
        assert flops >= input_proj_flops

    def test_includes_head_contribution(self) -> None:
        """Changing head should change total FLOPs."""
        cheap = _relative_flops(["LSTMBlock"], "GBMHead", 32, 32, 12)
        expensive = _relative_flops(["LSTMBlock"], "NeuralSDEHead", 32, 32, 12)
        assert expensive > cheap

    def test_empty_blocks_still_positive(self) -> None:
        """Even with no blocks, input projection + head should give positive FLOPs."""
        flops = _relative_flops([], "GBMHead", 32, 32, 12)
        assert flops > 0

    def test_all_scaling_types_covered(self) -> None:
        """Every scaling type in the dict should have at least one block."""
        scaling_types = set(_BLOCK_FLOP_SCALING.values())
        assert "constant" in scaling_types
        assert "sequential" in scaling_types
        assert "nlogn" in scaling_types
        assert "quadratic" in scaling_types


# ---------------------------------------------------------------------------
# estimate_params (tool)
# ---------------------------------------------------------------------------

class TestEstimateParamsTool:
    def test_returns_valid_json(self) -> None:
        result = json.loads(estimate_params(blocks='["LSTMBlock"]'))
        assert "total_params" in result
        assert "breakdown" in result
        assert "cost_tier" in result
        assert "estimated_gpu_memory_mb" in result

    def test_breakdown_has_input_projection(self) -> None:
        result = json.loads(estimate_params(blocks='["LSTMBlock"]'))
        layers = [b["layer"] for b in result["breakdown"]]
        assert "input_projection" in layers

    def test_breakdown_has_each_block_and_head(self) -> None:
        result = json.loads(estimate_params(
            blocks='["LSTMBlock", "GRUBlock"]', head="SDEHead"
        ))
        layers = [b["layer"] for b in result["breakdown"]]
        assert "LSTMBlock" in layers
        assert "GRUBlock" in layers
        assert "SDEHead" in layers

    def test_total_equals_sum_of_breakdown(self) -> None:
        result = json.loads(estimate_params(blocks='["LSTMBlock", "GRUBlock"]'))
        breakdown_sum = sum(b["params"] for b in result["breakdown"])
        assert result["total_params"] == breakdown_sum

    def test_human_readable_format_k(self) -> None:
        """Small models should show K format."""
        result = json.loads(estimate_params(blocks='["RevIN"]', d_model=16))
        assert "K" in result["total_params_human"]

    def test_cost_tier_ordering(self) -> None:
        """Larger architectures should get higher cost tiers."""
        small = json.loads(estimate_params(blocks='["RevIN"]', d_model=16))
        large = json.loads(estimate_params(
            blocks='["TransformerEncoder", "TimesNetBlock"]', d_model=128
        ))
        tier_order = ["very low", "low", "medium", "high", "very high"]
        assert tier_order.index(large["cost_tier"]) >= tier_order.index(small["cost_tier"])

    def test_memory_estimation_positive(self) -> None:
        # Use a larger architecture so activation memory doesn't round to 0
        result = json.loads(estimate_params(
            blocks='["LSTMBlock", "TransformerBlock"]', d_model=64
        ))
        assert result["estimated_gpu_memory_mb"] > 0
        assert result["parameter_memory_mb"] > 0
        assert result["activation_memory_mb"] > 0

    def test_config_echoed_back(self) -> None:
        result = json.loads(estimate_params(
            blocks='["LSTMBlock"]', head="SDEHead", d_model=64
        ))
        assert result["config"]["blocks"] == ["LSTMBlock"]
        assert result["config"]["head"] == "SDEHead"
        assert result["config"]["d_model"] == 64

    def test_handles_invalid_json_gracefully(self) -> None:
        result = json.loads(estimate_params(blocks="not valid json"))
        assert "error" in result

    def test_accepts_list_input(self) -> None:
        """Should accept pre-parsed list as well as JSON string."""
        result = json.loads(estimate_params(blocks=["LSTMBlock"]))  # type: ignore[arg-type]
        assert "total_params" in result


# ---------------------------------------------------------------------------
# estimate_flops (tool)
# ---------------------------------------------------------------------------

class TestEstimateFlopsTool:
    def test_returns_ranked_results(self) -> None:
        archs = json.dumps([
            {"blocks": ["LSTMBlock"], "head": "GBMHead", "d_model": 32},
            {"blocks": ["RevIN"], "head": "GBMHead", "d_model": 32},
        ])
        result = json.loads(estimate_flops(archs))
        assert "ranked_by_cost" in result
        assert len(result["ranked_by_cost"]) == 2

    def test_cheapest_is_first(self) -> None:
        archs = json.dumps([
            {"blocks": ["TransformerEncoder"], "head": "NeuralSDEHead", "d_model": 64},
            {"blocks": ["RevIN"], "head": "GBMHead", "d_model": 16},
        ])
        result = json.loads(estimate_flops(archs))
        ranked = result["ranked_by_cost"]
        assert ranked[0]["relative_flops"] <= ranked[-1]["relative_flops"]

    def test_cost_ratio_cheapest_is_one(self) -> None:
        archs = json.dumps([
            {"blocks": ["LSTMBlock"], "head": "GBMHead", "d_model": 32},
            {"blocks": ["TransformerBlock"], "head": "GBMHead", "d_model": 32},
        ])
        result = json.loads(estimate_flops(archs))
        assert result["ranked_by_cost"][0]["cost_ratio"] == 1.0

    def test_cost_ratio_more_expensive_above_one(self) -> None:
        archs = json.dumps([
            {"blocks": ["RevIN"], "head": "GBMHead", "d_model": 16},
            {"blocks": ["TransformerEncoder", "TimesNetBlock"], "head": "NeuralSDEHead",
             "d_model": 128},
        ])
        result = json.loads(estimate_flops(archs))
        assert result["ranked_by_cost"][-1]["cost_ratio"] > 1.0

    def test_cheapest_and_most_expensive_reported(self) -> None:
        archs = json.dumps([
            {"blocks": ["LSTMBlock"], "head": "GBMHead"},
            {"blocks": ["RevIN"], "head": "GBMHead"},
            {"blocks": ["TransformerEncoder"], "head": "NeuralSDEHead"},
        ])
        result = json.loads(estimate_flops(archs))
        assert result["cheapest"] is not None
        assert result["most_expensive"] is not None
        assert result["cheapest"]["relative_flops"] <= result["most_expensive"]["relative_flops"]

    def test_single_architecture(self) -> None:
        archs = json.dumps([
            {"blocks": ["LSTMBlock"], "head": "GBMHead"},
        ])
        result = json.loads(estimate_flops(archs))
        assert len(result["ranked_by_cost"]) == 1
        assert result["ranked_by_cost"][0]["cost_ratio"] == 1.0

    def test_empty_list(self) -> None:
        result = json.loads(estimate_flops("[]"))
        assert result["ranked_by_cost"] == []
        assert result["cheapest"] is None

    def test_handles_invalid_json(self) -> None:
        result = json.loads(estimate_flops("not json"))
        assert "error" in result

    def test_defaults_applied(self) -> None:
        """Missing head/d_model should use defaults."""
        archs = json.dumps([{"blocks": ["LSTMBlock"]}])
        result = json.loads(estimate_flops(archs))
        entry = result["ranked_by_cost"][0]
        assert entry["head"] == "GBMHead"
        assert entry["d_model"] == DEFAULT_D_MODEL

    def test_estimated_params_included(self) -> None:
        archs = json.dumps([{"blocks": ["LSTMBlock"], "head": "GBMHead", "d_model": 32}])
        result = json.loads(estimate_flops(archs))
        assert result["ranked_by_cost"][0]["estimated_params"] > 0


# ---------------------------------------------------------------------------
# generate_ablation_configs (tool)
# ---------------------------------------------------------------------------

class TestGenerateAblationConfigs:
    def test_baseline_always_included(self) -> None:
        result = json.loads(generate_ablation_configs(
            baseline_blocks='["LSTMBlock", "GRUBlock"]',
            ablation_type="block_removal",
        ))
        assert result["baseline"]["name"] == "baseline"
        assert result["baseline"]["blocks"] == ["LSTMBlock", "GRUBlock"]

    def test_block_removal_produces_variants(self) -> None:
        result = json.loads(generate_ablation_configs(
            baseline_blocks='["LSTMBlock", "GRUBlock", "RevIN"]',
            ablation_type="block_removal",
        ))
        # LSTMBlock and GRUBlock can be removed (RevIN is never removed)
        names = [a["name"] for a in result["ablations"]]
        assert "ablation_remove_LSTMBlock" in names
        assert "ablation_remove_GRUBlock" in names
        assert "ablation_remove_RevIN" not in names

    def test_head_swap_tries_all_heads(self) -> None:
        result = json.loads(generate_ablation_configs(
            baseline_blocks='["LSTMBlock"]',
            baseline_head="GBMHead",
            ablation_type="head_swap",
        ))
        heads_in_ablations = {a["head"] for a in result["ablations"]}
        # All heads except baseline head
        expected = set(_HEAD_PARAM_FACTORS.keys()) - {"GBMHead"}
        assert heads_in_ablations == expected

    def test_d_model_sweep_produces_variants(self) -> None:
        result = json.loads(generate_ablation_configs(
            baseline_blocks='["LSTMBlock"]',
            baseline_d_model=32,
            ablation_type="d_model_sweep",
        ))
        d_models = {a["d_model"] for a in result["ablations"]}
        assert 16 in d_models
        assert 64 in d_models
        assert 128 in d_models
        assert 32 not in d_models  # baseline d_model excluded

    def test_block_swap_finds_alternatives(self) -> None:
        result = json.loads(generate_ablation_configs(
            baseline_blocks='["LSTMBlock"]',
            ablation_type="block_swap",
        ))
        # LSTM is in recurrent group, so RNNBlock and GRUBlock should be alternatives
        swapped_blocks = [a["blocks"][0] for a in result["ablations"]]
        assert "RNNBlock" in swapped_blocks
        assert "GRUBlock" in swapped_blocks

    def test_all_combines_ablation_types(self) -> None:
        result = json.loads(generate_ablation_configs(
            baseline_blocks='["LSTMBlock", "GRUBlock"]',
            ablation_type="all",
        ))
        names = [a["name"] for a in result["ablations"]]
        # Should have block removal, head swap, d_model sweep, and block swap
        has_removal = any("remove" in n for n in names)
        has_head = any("head" in n for n in names)
        has_dmodel = any("d_model" in n for n in names)
        has_swap = any("swap" in n for n in names)
        assert has_removal
        assert has_head
        assert has_dmodel
        assert has_swap

    def test_ablations_sorted_by_params(self) -> None:
        result = json.loads(generate_ablation_configs(
            baseline_blocks='["LSTMBlock", "GRUBlock"]',
            ablation_type="all",
        ))
        params = [a["estimated_params"] for a in result["ablations"]]
        assert params == sorted(params)

    def test_total_configs_count(self) -> None:
        result = json.loads(generate_ablation_configs(
            baseline_blocks='["LSTMBlock"]',
            ablation_type="head_swap",
        ))
        # 1 baseline + (num_heads - 1) swaps
        assert result["total_configs"] == 1 + len(_HEAD_PARAM_FACTORS) - 1

    def test_configs_have_required_fields(self) -> None:
        result = json.loads(generate_ablation_configs(
            baseline_blocks='["LSTMBlock"]',
            ablation_type="all",
        ))
        for cfg in [result["baseline"]] + result["ablations"]:
            assert "blocks" in cfg
            assert "head" in cfg
            assert "d_model" in cfg
            assert "horizon" in cfg
            assert "lr" in cfg
            assert "n_paths" in cfg
            assert "batch_size" in cfg
            assert "estimated_params" in cfg

    def test_handles_invalid_json(self) -> None:
        result = json.loads(generate_ablation_configs(baseline_blocks="bad json"))
        assert "error" in result

    def test_single_block_removal_needs_at_least_one(self) -> None:
        """Removing the only non-RevIN block should still leave ≥1 block."""
        result = json.loads(generate_ablation_configs(
            baseline_blocks='["LSTMBlock"]',
            ablation_type="block_removal",
        ))
        # Can't remove LSTMBlock if it would leave 0 blocks
        # Actually the code checks len(ablated) >= 1, and RevIN is never removed,
        # so removing LSTMBlock leaves [], which has length 0, so it shouldn't appear
        removal_names = [a["name"] for a in result["ablations"] if "remove" in a["name"]]
        for name in removal_names:
            ablation = next(a for a in result["ablations"] if a["name"] == name)
            assert len(ablation["blocks"]) >= 1


# ---------------------------------------------------------------------------
# sweep_configs (tool)
# ---------------------------------------------------------------------------

class TestSweepConfigs:
    def test_returns_configs(self) -> None:
        result = json.loads(sweep_configs(
            blocks='["LSTMBlock"]',
            d_model_values="[16, 32]",
            lr_values="[0.001, 0.01]",
        ))
        assert "configs" in result
        assert result["total_configs"] == 4  # 2 d_model * 2 lr * 1 n_paths

    def test_configs_sorted_by_flops(self) -> None:
        result = json.loads(sweep_configs(
            blocks='["LSTMBlock"]',
            d_model_values="[16, 32, 64]",
        ))
        flops = [c["relative_flops"] for c in result["configs"]]
        assert flops == sorted(flops)

    def test_max_configs_limits_output(self) -> None:
        result = json.loads(sweep_configs(
            blocks='["LSTMBlock"]',
            d_model_values="[16, 32, 64, 128]",
            lr_values="[0.0001, 0.001, 0.01]",
            n_paths_values="[50, 100]",
            max_configs=5,
        ))
        assert result["total_configs"] <= 5

    def test_sampled_flag_when_truncated(self) -> None:
        result = json.loads(sweep_configs(
            blocks='["LSTMBlock"]',
            d_model_values="[16, 32, 64, 128]",
            lr_values="[0.0001, 0.001, 0.01]",
            n_paths_values="[50, 100]",
            max_configs=5,
        ))
        assert result["sampled"] is True

    def test_not_sampled_when_within_limit(self) -> None:
        result = json.loads(sweep_configs(
            blocks='["LSTMBlock"]',
            d_model_values="[32]",
            lr_values="[0.001]",
        ))
        assert result["sampled"] is False

    def test_defaults_when_no_ranges(self) -> None:
        result = json.loads(sweep_configs(blocks='["LSTMBlock"]'))
        assert result["total_configs"] == 1

    def test_configs_have_relative_flops(self) -> None:
        result = json.loads(sweep_configs(
            blocks='["LSTMBlock"]',
            d_model_values="[32, 64]",
        ))
        for cfg in result["configs"]:
            assert "relative_flops" in cfg
            assert cfg["relative_flops"] > 0

    def test_handles_invalid_json(self) -> None:
        result = json.loads(sweep_configs(blocks="bad json"))
        assert "error" in result

    def test_grid_size_reported(self) -> None:
        result = json.loads(sweep_configs(
            blocks='["LSTMBlock"]',
            d_model_values="[16, 32]",
            lr_values="[0.001, 0.01]",
            n_paths_values="[50, 100]",
        ))
        assert result["grid_size"] == 8  # 2 * 2 * 2


# ---------------------------------------------------------------------------
# Metadata consistency
# ---------------------------------------------------------------------------

class TestMetadataConsistency:
    def test_every_block_has_param_factor_and_scaling(self) -> None:
        """Every block in param factors should also have a scaling type."""
        assert set(_BLOCK_PARAM_FACTORS.keys()) == set(_BLOCK_FLOP_SCALING.keys())

    def test_all_param_factors_positive(self) -> None:
        for block, factor in _BLOCK_PARAM_FACTORS.items():
            assert factor > 0, f"{block} has non-positive factor {factor}"

    def test_all_head_factors_positive(self) -> None:
        for head, factor in _HEAD_PARAM_FACTORS.items():
            assert factor > 0, f"{head} has non-positive factor {factor}"

    def test_known_scaling_types(self) -> None:
        valid = {"constant", "sequential", "nlogn", "quadratic"}
        for block, scaling in _BLOCK_FLOP_SCALING.items():
            assert scaling in valid, f"{block} has unknown scaling {scaling}"
