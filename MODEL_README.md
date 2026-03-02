---
language: en
license: mit
library_name: open-synth-miner
tags:
  - time-series
  - probabilistic-forecasting
  - monte-carlo
  - bittensor
  - sn50
  - synth
  - crps
  - price-forecasting
  - hybrid-architecture
  - pytorch
datasets:
  - tensorlink-dev/open-synth-training-data
pipeline_tag: time-series-forecasting
---

# PAG — Hybrid SN50 Probabilistic Price Forecaster

**PAG** is a hybrid probabilistic price forecasting model built for [Bittensor Subnet 50 (Synth)](https://bittensor.com/). It generates Monte Carlo price path simulations scored by **CRPS** (Continuous Ranked Probability Score) — a metric that rewards well-calibrated probability distributions, not just point accuracy.

Built with [open-synth-miner](https://github.com/tensorlink-dev/open-synth-miner) and trained/published via [synth-city](https://github.com/tensorlink-dev/synth-city).

## Model Description

PAG uses a **hybrid backbone architecture** — stacking multiple neural network paradigms (e.g. attention + recurrence, convolution + frequency analysis) into a single composable pipeline. All backbone blocks share a uniform tensor interface:

```
(batch, seq, d_model) → (batch, seq, d_model)
```

This lets blocks be freely combined and swapped. The hybrid approach captures different aspects of price dynamics — long-range dependencies via attention, sequential momentum via recurrence, local patterns via convolution — and fuses them into a single learned representation before the prediction head generates full probabilistic forecasts.

### Architecture

| Component | Description |
|-----------|-------------|
| **Framework** | [open-synth-miner](https://github.com/tensorlink-dev/open-synth-miner) — composable PyTorch framework |
| **Backbone** | Hybrid multi-block (stacked heterogeneous blocks with uniform tensor interface) |
| **Prediction Head** | Generates drift (μ) and volatility (σ) parameters for Monte Carlo simulation |
| **Normalization** | RevIN (Reversible Instance Normalization) as first block |
| **Output** | 1,000 Monte Carlo price paths per asset per timeframe |

### Supported Blocks

The open-synth-miner framework provides 15 backbone blocks that can be composed in any order:

| Block | Cost | Best For |
|-------|------|----------|
| RevIN | Very low | Input normalization (must be first) |
| LayerNormBlock | Very low | Inter-block normalization |
| DLinearBlock | Very low | Decomposition baseline |
| RNNBlock | Low | Minimal recurrence |
| ResConvBlock | Low | Local feature extraction |
| BiTCNBlock | Low | Dilated temporal convolution |
| SDEEvolutionBlock | Low | Stochastic differential equation residual |
| GRUBlock | Low-Med | Gated recurrent (lighter LSTM alternative) |
| LSTMBlock | Medium | Sequential and momentum patterns |
| FourierBlock | Medium | Periodic/frequency-domain patterns |
| TransformerBlock | Medium | Long-range self-attention |
| TimeMixerBlock | Medium | Multi-scale temporal mixing |
| Unet1DBlock | Medium | Multi-resolution features |
| TransformerEncoder | High | Deep multi-head attention |
| TimesNetBlock | High | Period-aware 2D convolution |

### Prediction Heads

| Head | Expressiveness | Description |
|------|---------------|-------------|
| GBMHead | Low | Geometric Brownian Motion — constant μ, σ |
| SDEHead | Medium | Deeper μ, σ networks |
| SimpleHorizonHead | Medium | Per-step prediction via pooling |
| HorizonHead | High | Per-step via cross-attention |
| NeuralBridgeHead | High | Macro + micro hierarchy |
| NeuralSDEHead | Very High | Full neural SDE |

## Intended Use

This model is designed for **Bittensor Subnet 50 (Synth)** — a decentralized competition where miners submit probabilistic price forecasts and are scored on CRPS.

### Prediction Targets

The model forecasts price paths for **9 assets** across **two timeframes**:

**Standard (24h)** — 288 steps at 5-minute intervals

**HFT (1h)** — 60 steps at 1-minute intervals

| Asset | Scoring Weight | Description |
|-------|---------------|-------------|
| BTC | 1.00 | Bitcoin |
| ETH | 0.67 | Ethereum |
| SOL | 0.59 | Solana |
| XAU | 2.26 | Gold |
| SPYX | 2.99 | S&P 500 |
| NVDAX | 1.39 | NVIDIA |
| TSLAX | 1.42 | Tesla |
| AAPLX | 1.86 | Apple |
| GOOGLX | 1.43 | Alphabet/Google |

Higher-weighted assets (SPYX, XAU, AAPLX) have more impact on overall miner ranking.

### Output Format

Each prediction produces a tensor of shape:

```
(n_paths, horizon_steps) = (1000, 288) for standard timeframe
(n_paths, horizon_steps) = (1000, 60)  for HFT timeframe
```

Each value represents a simulated future price at that time step along one Monte Carlo path.

## Evaluation

### CRPS (Continuous Ranked Probability Score)

Models are scored using CRPS — **lower is better**:

```
CRPS = (1/N) × Σ|yₙ - x| - (1/2N²) × Σₙ Σₘ |yₙ - yₘ|
```

Where:
- `yₙ` = forecast values (one per Monte Carlo path)
- `x` = realized/observed price
- **Term 1**: Mean absolute error between ensemble and observation
- **Term 2**: Mean pairwise absolute difference (rewards calibration and diversity)

CRPS is evaluated at multiple time horizons: **5, 10, 15, 30, 60, 180, 360, 720, and 1440 minutes**.

### Why CRPS?

Unlike simple point forecast metrics (MAE, RMSE), CRPS evaluates the **entire predicted distribution**. A model can't game CRPS by just predicting the mean — it must produce Monte Carlo paths that genuinely capture the range and shape of possible future prices. This rewards:

- **Calibration** — predicted uncertainty matches realized uncertainty
- **Sharpness** — distributions are as tight as possible while remaining calibrated
- **Tail coverage** — extreme moves are represented in the ensemble

## Training

### Data

Training data sourced from [tensorlink-dev/open-synth-training-data](https://huggingface.co/datasets/tensorlink-dev/open-synth-training-data) on Hugging Face — historical OHLCV price data for all 9 SN50 assets.

| Parameter | Value |
|-----------|-------|
| **Input features** | 4 (OHLCV) |
| **Sequence length (5m)** | 288 (24h lookback at 5-min intervals) |
| **Prediction horizon (5m)** | 288 (24h forecast at 5-min intervals) |
| **Sequence length (1m)** | 60 (1h lookback at 1-min intervals) |
| **Prediction horizon (1m)** | 60 (1h forecast at 1-min intervals) |
| **Monte Carlo paths** | 1,000 (production) |
| **Feature engineering** | ZScore normalization |

### Compute

Training runs on decentralized GPU infrastructure via **Basilica (Bittensor SN39)** — a GPU compute marketplace providing Tesla V100, RTX-A4000, and RTX-A6000 GPUs.

### Pipeline

The model was discovered, trained, validated, and published by [synth-city](https://github.com/tensorlink-dev/synth-city)'s autonomous agent pipeline:

1. **Planner** — surveyed available blocks/heads and past experiment history to design the architecture
2. **Trainer** — executed the experiment on decentralized GPUs
3. **CodeChecker** — validated the configuration and output tensors against SN50 requirements
4. **Debugger** — diagnosed and fixed any training failures
5. **Publisher** — published to Hugging Face Hub after confirming CRPS improvement over prior models

## How to Use

### With open-synth-miner

```python
from osa.models.factory import create_model
from osa.models.registry import discover_components
from omegaconf import OmegaConf

# Discover all registered components
discover_components("src/models/components")

# Load experiment config (adjust blocks/head to match your published model)
config = OmegaConf.create({
    "model": {
        "backbone": {
            "blocks": ["RevIN", "TransformerBlock", "LSTMBlock"],
            "d_model": 32,
            "feature_dim": 4,
            "seq_len": 288,
        },
        "head": {
            "_target_": "GBMHead"
        }
    },
    "training": {
        "horizon": 288,
        "n_paths": 1000,
    }
})

model = create_model(config)
```

### With synth-city CLI

```bash
# Run a quick experiment with a hybrid architecture
synth-city experiment --blocks TransformerBlock,LSTMBlock --head GBMHead --epochs 5

# Or let the autonomous pipeline find the best hybrid architecture
synth-city pipeline --publish
```

### Loading from Hugging Face Hub

```python
from osa.tracking.hub_manager import HubManager

# Load published model weights
manager = HubManager(repo_id="tensorlink-dev/pag-hybrid-sn50")
model = manager.load_model()
```

## Technical Details

### Tensor Interface

Every backbone block in open-synth-miner adheres to a strict uniform interface:

```
Input:  (batch_size, sequence_length, d_model)
Output: (batch_size, sequence_length, d_model)
```

This enables arbitrary block composition — any block can follow any other block. The hybrid approach exploits this by stacking blocks from different architectural families (attention, recurrence, convolution, frequency analysis) to capture complementary patterns in price data.

### Monte Carlo Simulation

The prediction head outputs drift (μ) and volatility (σ) parameters at each time step. These parameterize a stochastic process from which 1,000 Monte Carlo paths are sampled, producing a full probabilistic forecast rather than a single point estimate.

### SN50 Validation

Before submission, outputs are validated against Subnet 50 requirements:
- Correct number of paths (1,000)
- Correct horizon lengths (288 for 5m, 60 for 1m)
- Finite, positive price values
- Valid tensor shapes

## Citation

If you use this model or the underlying framework, please reference:

- [open-synth-miner](https://github.com/tensorlink-dev/open-synth-miner) — composable PyTorch framework for probabilistic forecasting
- [synth-city](https://github.com/tensorlink-dev/synth-city) — agentic R&D and MLOps engine for Bittensor mining
- [Bittensor Subnet 50 (Synth)](https://bittensor.com/) — decentralized price forecasting competition

## License

MIT
