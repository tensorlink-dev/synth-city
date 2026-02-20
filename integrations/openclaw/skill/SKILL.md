---
name: synth-city
description: >-
  Autonomous AI research pipeline for Bittensor Subnet 50 (Synth).
  Discovers, trains, debugs, and publishes probabilistic price forecasting models.
metadata:
  clawdbot:
    requires:
      bins:
        - curl
        - python
      env:
        - CHUTES_API_KEY
    primaryEnv: CHUTES_API_KEY
    homepage: https://github.com/tensorlink-dev/synth-city
---

# synth-city — Bittensor SN50 Research Assistant

You have access to **synth-city**, an agentic pipeline for competitive probabilistic price forecasting on Bittensor Subnet 50 (Synth). Use this skill to let the user manage experiments, run the research pipeline, check market data, review historical results, and monitor model performance — all from chat.

## What synth-city does

synth-city discovers neural network components (backbone blocks + prediction heads), plans experiment architectures, trains models, validates results against SN50 specifications, and publishes the best models.

The scoring metric is **CRPS** (Continuous Ranked Probability Score) — lower is better. CRPS rewards well-calibrated probability distributions, not just accurate point forecasts. Sharpness and log-likelihood are diagnostic metrics only; CRPS is the only one that affects SN50 miner ranking.

### Pipeline stages

The full pipeline chains 4-5 agents together:

1. **Planner** — discovers available components, reviews prior results, produces an experiment plan
2. **Trainer** — executes experiments from the plan, reports the best result
3. **CodeChecker** — validates the best experiment config and output shapes
4. **Debugger** — fixes failed experiments (only runs if CodeChecker fails)
5. **Publisher** — publishes the winning model to HF Hub + W&B (optional)

The pipeline retries failed stages with escalating temperature (0.1 → 0.2 → 0.3...) and detects stalls when the debugger produces identical configs across attempts.

## Bridge API

All commands go through the synth-city bridge server. The bridge must be running on the same machine as the OpenClaw gateway (default: `http://127.0.0.1:8377`).

## Available commands

When the user asks about any of the topics below, call the bridge API using the appropriate endpoint.

### Pipeline control

- **"run the pipeline"** / **"start research"** / **"find better models"**
  `POST http://127.0.0.1:8377/pipeline/run`
  Body: `{"channel": "default", "retries": 5, "temperature": 0.1, "publish": false}`
  The pipeline runs asynchronously. Tell the user it has started and they can check status.
  Set `"publish": true` to auto-publish the best model to HF Hub when the pipeline completes.

- **"pipeline status"** / **"how's the run going"**
  `GET http://127.0.0.1:8377/pipeline/status`
  Returns running state, current stage, elapsed time, and results when complete.

### Component discovery

- **"what blocks are available"** / **"list components"**
  `GET http://127.0.0.1:8377/components/blocks`
  Returns all 15 backbone blocks with parameters, strengths, and compute cost.

- **"what heads can I use"**
  `GET http://127.0.0.1:8377/components/heads`
  Returns all 6 head types with parameters and expressiveness levels.

- **"show me the presets"** / **"what combinations are ready to go"**
  `GET http://127.0.0.1:8377/components/presets`
  Returns 10 ready-to-run block+head combinations.

### Experiment management

- **"create an experiment with Transformer and LSTM"**
  `POST http://127.0.0.1:8377/experiment/create`
  Body: `{"blocks": ["TransformerBlock", "LSTMBlock"], "head": "GBMHead", "d_model": 32}`
  Optional fields: `horizon` (default 12), `n_paths` (default 100), `lr` (default 0.001), `seq_len` (default 32), `batch_size` (default 4), `feature_dim` (default 4).

- **"validate my experiment"** / **"check this config"**
  `POST http://127.0.0.1:8377/experiment/validate`
  Body: `{"experiment": <config_json>}`
  Returns param count, errors, and warnings without running the experiment.

- **"run this experiment"**
  `POST http://127.0.0.1:8377/experiment/run`
  Body: `{"experiment": <config_json>, "epochs": 1, "name": "optional-label"}`
  Trains the model and returns CRPS, sharpness, and log-likelihood metrics.

- **"compare results"** / **"show rankings"** / **"what's the best so far"**
  `GET http://127.0.0.1:8377/experiment/compare`
  Returns all experiments from this session ranked by CRPS (best first).

- **"session summary"** / **"what have we tried"**
  `GET http://127.0.0.1:8377/session/summary`
  Returns count of experiments, current best, and full result list.

- **"clear session"** / **"reset experiments"** / **"start fresh"**
  `POST http://127.0.0.1:8377/session/clear`
  Resets in-memory results. Use between unrelated experiment batches.

### Market data

- **"what's the BTC price"** / **"ETH price"**
  `GET http://127.0.0.1:8377/market/price/BTC`
  Live price from the Pyth oracle. Supported assets: BTC, ETH, SOL, XAU, SPYX, NVDAX, TSLAX, AAPLX, GOOGLX.

- **"show me SOL history"** / **"ETH price last 7 days"**
  `GET http://127.0.0.1:8377/market/history/SOL?days=30`
  Historical OHLCV data. Default 30 days.

### Component registry (add new blocks / heads)

- **"what component files exist"** / **"list registry files"**
  `GET http://127.0.0.1:8377/registry/files`
  Lists all Python files in the open-synth-miner component directory.

- **"show me the transformer block code"** / **"read a component"**
  `GET http://127.0.0.1:8377/registry/read?path=src/models/components/transformer.py`
  Returns the source code of a component file. Use this to study existing blocks before writing new ones.

- **"write a new block"** / **"add a wavelet block"**
  `POST http://127.0.0.1:8377/registry/write`
  Body: `{"filename": "wavelet_block.py", "code": "<full Python source>"}`
  Writes a new PyTorch block or head into `src/models/components/`. The component must follow the uniform tensor interface: `(batch, seq, d_model) → (batch, seq, d_model)` for blocks, or appropriate output shape for heads. **Always call reload after writing.**

- **"reload the registry"** / **"refresh components"**
  `POST http://127.0.0.1:8377/registry/reload`
  Body: `{}`
  Re-discovers components from disk so newly written blocks/heads appear in `list_blocks` / `list_heads` immediately.

### HF Hub (download models / retrieve URLs)

- **"list published models"** / **"what's on HF Hub"**
  `GET http://127.0.0.1:8377/hf/models`
  Optional: `?repo_id=org/repo` to query a specific repo (defaults to the configured repo).
  Returns files, branches, version tags, download counts, and metadata. Use the `repo_id` and branch names to construct download URLs like `https://huggingface.co/{repo_id}/resolve/{branch}/{filename}`.

- **"show the model card"** / **"what's the latest model about"**
  `GET http://127.0.0.1:8377/hf/model-card`
  Optional: `?repo_id=org/repo&revision=main`
  Returns the README/model card content, structured metadata, and config.json if present.

- **"download experiment config from HF"**
  `GET http://127.0.0.1:8377/hf/artifact?filename=experiment.json`
  Optional: `&repo_id=org/repo&revision=main`
  Downloads and returns a JSON artifact from the HF repo.

### History (query tested models)

- **"list all pipeline runs"** / **"what runs do we have"**
  `GET http://127.0.0.1:8377/history/runs`
  Lists all pipeline runs stored in Hippius with run IDs and file manifests. Most recent first.

- **"show me run X"** / **"load the latest run"**
  `GET http://127.0.0.1:8377/history/run/{run_id}`
  Loads summary, CRPS ranking, and individual experiments for a specific run. Use `latest` as the run_id to load the most recent run.

- **"what architectures have been tested"** / **"best experiments so far"**
  `GET http://127.0.0.1:8377/history/experiments?limit=50`
  Returns the best experiments across **all** pipeline runs, sorted by CRPS. Each entry includes block composition, head type, and full metrics. Use this to avoid re-testing architectures that have already been tried.

- **"show W&B results"** / **"what does wandb say"**
  `GET http://127.0.0.1:8377/history/wandb?limit=20&order=best`
  Fetches runs from Weights & Biases. Order: `best` (lowest CRPS), `recent` (newest), or `worst`. Returns configs, metrics, tags, and W&B URLs.

## Component quick reference

Use this to help the user pick architectures.

### Blocks (15 available)

All blocks share the same tensor interface: `(batch, seq, d_model) → (batch, seq, d_model)`, so any block can be stacked with any other.

| Name | Cost | Best for |
|------|------|----------|
| RevIN | very low | Input normalization — **must be first** if used |
| LayerNormBlock | very low | Inter-block normalization |
| DLinearBlock | very low | Decomposition baseline |
| RNNBlock | low | Minimal recurrence |
| ResConvBlock | low | Local features |
| BiTCNBlock | low | Dilated local patterns |
| SDEEvolutionBlock | low | Stochastic residual |
| GRUBlock | low-med | Lighter LSTM alternative |
| LSTMBlock | medium | Sequential/momentum patterns |
| FourierBlock | medium | Periodic patterns |
| TransformerBlock | medium | Long-range attention |
| TimeMixerBlock | medium | Multi-scale mixing |
| Unet1DBlock | medium | Multi-resolution |
| TransformerEncoder | high | Deep attention |
| TimesNetBlock | high | Period-aware 2D convolution |

### Heads (6 available)

| Name | Expressiveness |
|------|----------------|
| GBMHead | Low — constant drift/vol, simplest baseline |
| SDEHead | Medium — deeper drift/vol network |
| SimpleHorizonHead | Medium — per-step via pooling |
| HorizonHead | High — per-step via cross-attention |
| NeuralBridgeHead | High — macro+micro hierarchy |
| NeuralSDEHead | Very high — full neural SDE |

### Composition rules

- `d_model` must be divisible by `nhead` (default nhead=4 → use 16, 32, 64, or 128)
- `RevIN` must be the **first** block if used
- `LayerNormBlock` goes **between** other blocks
- Deeper stacks (3-4 blocks) need `d_model >= 32`
- Hybrid stacks (e.g. Transformer + LSTM) often outperform single-block architectures
- `GBMHead` is the simplest — upgrading to `SDEHead` or `NeuralSDEHead` often improves CRPS significantly
- Don't over-parameterise: `d_model=32` is often sufficient; `d_model=128` can overfit

## Response formatting

- Format CRPS scores to 6 decimal places.
- List components as clean bulleted lists with name, cost/expressiveness, and description.
- Show experiment rankings as numbered lists sorted by CRPS (best first).
- For pipeline status, give a concise summary: current stage, elapsed time, success/failure.
- For prices, show the asset name and price with appropriate precision.
- When comparing experiments, highlight what changed between configs (blocks, head, d_model) alongside the CRPS difference.

## SN50 context

The Synth subnet (SN50) requires miners to produce **1,000 Monte Carlo price paths** per asset at 5-minute intervals over a 24-hour horizon (289 timesteps). Predictions are scored using CRPS.

### Supported assets (with scoring weights)

| Asset | Weight | Description |
|-------|--------|-------------|
| BTC | 1.00 | Bitcoin |
| ETH | 0.67 | Ethereum |
| SOL | 0.59 | Solana |
| XAU | 2.26 | Gold |
| SPYX | 2.99 | S&P 500 |
| NVDAX | 1.39 | NVIDIA |
| TSLAX | 1.42 | Tesla |
| AAPLX | 1.86 | Apple |
| GOOGLX | 1.43 | Alphabet |

Higher-weighted assets (SPYX, XAU, AAPLX) have more impact on the overall score. Guide the user toward architectures that perform well on these high-weight assets.

### Research vs production

- Research mode uses `n_paths=100` and small `d_model` for fast iteration.
- Production submission needs `n_paths=1000` and the full 24-hour horizon.
- Always validate experiments before publishing.

## Error handling

- If the bridge returns `{"error": "..."}`, report the error clearly and suggest the user check that the bridge is running (`python main.py bridge`).
- If pipeline status shows `"status": "failed"`, show the error and suggest rerunning with higher retries or different parameters.
- If an experiment returns a CRPS of `null` or very high values (> 1.0), the model likely failed to converge — suggest trying a simpler architecture or lower learning rate.
- If HF Hub or W&B endpoints return errors, verify that `HF_REPO_ID` / `WANDB_PROJECT` are configured in the `.env` file and that API tokens are set.
- After writing a new component, **always** call the reload endpoint before attempting to use the new block/head in experiments. If reload fails, the component source likely has an import error — read it back and fix.
- If Hippius history returns empty results, it means no experiments have been persisted to decentralised storage yet. Run the pipeline or manually save experiments first.
