# Configuration

All synth-city settings are managed through environment variables, loaded from a `.env` file via [python-dotenv](https://pypi.org/project/python-dotenv/). The configuration module lives in `config.py`.

To get started, copy the example file and fill in your values:

```bash
cp .env.example .env
```

---

## Quick Reference

Only the Chutes AI API key is strictly required to run the pipeline. Everything else has sensible defaults or is optional depending on which features you use.

| What you want to do | Required variables |
|---------------------|--------------------|
| Run the pipeline (basic) | `CHUTES_API_KEY` |
| Train on remote GPUs | `BASILICA_API_TOKEN` |
| Persist experiment history | `HIPPIUS_ENDPOINT`, `HIPPIUS_ACCESS_KEY`, `HIPPIUS_SECRET_KEY` |
| Publish models | `HF_TOKEN`, `HF_REPO_ID` |
| Mine on SN50 | `BT_WALLET_NAME`, `BT_HOTKEY_NAME` |
| Run the bridge server | No extra config (defaults to localhost:8377) |

---

## LLM Inference (Chutes AI)

Connection to the LLM provider that powers all agent reasoning. Any OpenAI-compatible endpoint works.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CHUTES_API_KEY` | `str` | — | API key for Chutes AI (required) |
| `CHUTES_BASE_URL` | `str` | `https://llm.chutes.ai/v1` | Base URL for the LLM endpoint |
| `DEFAULT_MODEL` | `str` | `Qwen/Qwen2.5-Coder-32B-Instruct` | Default model for all agents |

The client (`pipeline/providers/chutes_client.py`) uses the OpenAI Python SDK with these timeouts: 120s connect, 300s read, 60s write, 60s pool. The SDK retries transient errors 3 times automatically; an outer retry wrapper adds 4 more attempts with exponential backoff.

To use a different LLM provider (OpenRouter, vLLM, Ollama, Together AI), change `CHUTES_BASE_URL` and `CHUTES_API_KEY` to match your provider's endpoint and credentials.

---

## Per-Agent Model Selection

Each agent can use a different model. Larger reasoning models are recommended for agents that make complex decisions (Planner, Trainer, Author); smaller code-tuned models work well for mechanical tasks (CodeChecker, Debugger, Publisher).

| Variable | Default | Agent |
|----------|---------|-------|
| `PLANNER_MODEL` | `Qwen/Qwen3-235B-A22B` | Planner |
| `TRAINER_MODEL` | `Qwen/Qwen3-235B-A22B` | Trainer |
| `AUTHOR_MODEL` | `Qwen/Qwen3-235B-A22B` | Component Author |
| `CODECHECKER_MODEL` | `Qwen/Qwen2.5-Coder-32B-Instruct` | CodeChecker |
| `DEBUGGER_MODEL` | `Qwen/Qwen2.5-Coder-32B-Instruct` | Debugger |
| `PUBLISHER_MODEL` | `Qwen/Qwen2.5-Coder-32B-Instruct` | Publisher |

If an agent-specific variable is not set, the agent falls back to `DEFAULT_MODEL`. The resolution logic is in `config.py:model_for(agent_name)`:

```python
from config import model_for

model = model_for("planner")  # returns PLANNER_MODEL or DEFAULT_MODEL
```

Override for a single run without changing `.env`:

```bash
PLANNER_MODEL=gpt-4 synth-city pipeline
```

---

## Research Defaults

Default hyperparameters for experiments created through the tool system. These are deliberately small for fast iteration during research. Increase them for production-quality results.

| Variable | Type | Default | Production | Description |
|----------|------|---------|------------|-------------|
| `RESEARCH_N_PATHS` | `int` | `100` | `1000` | Monte Carlo paths per experiment |
| `RESEARCH_D_MODEL` | `int` | `32` | `64–128` | Hidden dimension of backbone blocks |
| `RESEARCH_HORIZON` | `int` | `12` | `288` | Prediction steps (at 5-min intervals) |
| `RESEARCH_SEQ_LEN` | `int` | `32` | `64–128` | Input sequence length |
| `RESEARCH_FEATURE_DIM` | `int` | `4` | `4` | Number of input features (OHLC) |
| `RESEARCH_BATCH_SIZE` | `int` | `4` | `16–64` | Training batch size |
| `RESEARCH_LR` | `float` | `0.001` | `0.0001–0.001` | Learning rate |
| `RESEARCH_EPOCHS` | `int` | `1` | `5+` | Training epochs per experiment |

---

## Timeframe Configuration

Predefined configurations for the two supported prediction timeframes.

| Timeframe | `pred_len` | `input_len` | File Suffix | Description |
|-----------|-----------|-------------|-------------|-------------|
| `5m` | `288` | `288` | `5m.parquet` | 24-hour horizon at 5-min intervals |
| `1m` | `60` | `60` | `1m.parquet` | 1-hour horizon at 1-min intervals |

These are accessed via `config.TIMEFRAME_CONFIGS["5m"]` and used by `create_experiment()` when a timeframe is specified.

---

## Data Loader

Settings for the market data loader that feeds training data to experiments.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HF_TRAINING_DATA_REPO` | `str` | `tensorlink-dev/open-synth-training-data` | HuggingFace dataset repository |
| `DATA_BATCH_SIZE` | `int` | `64` | Loader batch size |
| `DATA_STRIDE` | `int` | `12` | Stride for sliding window |
| `DATA_GAP_HANDLING` | `str` | `ffill` | Gap handling strategy (forward fill) |
| `DATA_FEATURE_ENGINEER` | `str` | `zscore` | Feature engineering method |

### Asset Name Mapping

SN50 asset names map to HuggingFace dataset names via `SN50_TO_HF_ASSET`:

| SN50 Name | HF Dataset Name |
|-----------|-----------------|
| `BTC` | `BTC_USD` |
| `ETH` | `ETH_USD` |
| `SOL` | `SOL_USD` |
| `SPYX` | `SPY` |
| `NVDAX` | `NVDA` |
| `TSLAX` | `TSLA` |
| `AAPLX` | `AAPL` |
| `GOOGLX` | `GOOGL` |

---

## SN50 Assets and Scoring

The target subnet configuration. These are constants, not user-configurable.

| Variable | Value | Description |
|----------|-------|-------------|
| `SN50_NUM_PATHS` | `1000` | Monte Carlo paths per submission |
| `SN50_HORIZON_MINUTES` | `1440` | 24-hour prediction horizon |
| `SN50_STEP_MINUTES` | `5` | Timestep interval |
| `SN50_HFT_HORIZON_MINUTES` | `60` | 1-hour HFT horizon |

### Asset Weights

```python
SN50_ASSETS = {
    "BTC": 1.00,  "ETH": 0.67,   "SOL": 0.59,
    "XAU": 2.26,  "SPYX": 2.99,  "NVDAX": 1.39,
    "TSLAX": 1.42, "AAPLX": 1.86, "GOOGLX": 1.43,
}
```

Higher weights mean the asset contributes more to overall score. SPYX (2.99) and XAU (2.26) are the most impactful.

### Subnet Constants (subnet/config.py)

| Constant | Value | Description |
|----------|-------|-------------|
| `NUM_STEPS_24H` | `289` | 24h steps including t0 |
| `NUM_STEPS_HFT` | `13` | 1h steps including t0 |
| `CRPS_EVAL_INCREMENTS` | `[5,10,15,30,60,180,360,720,1440]` | CRPS evaluation horizons (minutes) |
| `EMISSION_SOFTMAX_BETA` | `-0.1` | Emission allocation parameter |
| `LEADERBOARD_WINDOW_DAYS` | `10` | Rolling leaderboard window |
| `SCORE_CAP_PERCENTILE` | `0.90` | Score cap for outlier protection |

---

## Basilica GPU Cloud (SN39)

Configuration for decentralized GPU compute via the Basilica marketplace.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BASILICA_API_TOKEN` | `str` | — | API token (falls back to `BASILICA_API_KEY`) |
| `BASILICA_API_URL` | `str` | `https://api.basilica.ai` | Basilica API endpoint |
| `BASILICA_MAX_HOURLY_RATE` | `float` | `0.44` | Maximum GPU rental rate (USD/hr) |
| `BASILICA_ALLOWED_GPU_TYPES` | `list[str]` | `TESLA V100, RTX-A4000, RTX-A6000` | Allowed GPU types (comma-separated) |

### Deployment Settings

Settings for Docker-based GPU pod deployments.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BASILICA_DEPLOY_IMAGE` | `str` | `ghcr.io/tensorlink-ai/synth-city-gpu:latest` | Docker image for training pods |
| `BASILICA_DEPLOY_GPU_MODELS` | `list[str]` | (same as allowed types) | GPU models for deployments |
| `BASILICA_DEPLOY_MIN_GPU_MEMORY_GB` | `int` | `12` | Minimum GPU memory |
| `BASILICA_DEPLOY_CPU` | `str` | `2000m` | Pod CPU request |
| `BASILICA_DEPLOY_MEMORY` | `str` | `8Gi` | Pod memory request |

---

## Bittensor Wallet

Configuration for the Bittensor wallet used for SN50 mining.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BT_WALLET_NAME` | `str` | `default` | Wallet cold key name |
| `BT_HOTKEY_NAME` | `str` | `default` | Wallet hot key name |
| `BT_NETWORK` | `str` | `finney` | Bittensor network (finney = mainnet) |
| `BT_NETUID` | `int` | `50` | Subnet UID (50 = Synth) |

---

## Market Data

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PYTH_PRICE_FEED_URL` | `str` | `https://hermes.pyth.network` | Pyth Network oracle endpoint |

The Pyth endpoint works without authentication. It provides real-time price feeds for BTC, ETH, SOL, and XAU.

---

## Publishing

Configuration for model publishing to HuggingFace Hub and experiment tracking.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HF_TOKEN` | `str` | — | HuggingFace API token |
| `HF_REPO_ID` | `str` | — | HuggingFace repository ID (e.g., `user/model-repo`) |
| `WANDB_PROJECT` | `str` | `synth-city` | Weights & Biases project name |
| `TRACKIO_PROJECT` | `str` | `synth-city` | Trackio project name |

Before publishing, authenticate with each service:

```bash
huggingface-cli login       # HuggingFace
wandb login                 # Weights & Biases (optional)
```

---

## Hippius Decentralized Storage (SN75)

S3-compatible storage for persisting experiment history across runs. Any S3-compatible endpoint works (Hippius, Cloudflare R2, MinIO, AWS S3).

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HIPPIUS_ENDPOINT` | `str` | — | S3 endpoint URL |
| `HIPPIUS_ACCESS_KEY` | `str` | — | S3 access key |
| `HIPPIUS_SECRET_KEY` | `str` | — | S3 secret key |
| `HIPPIUS_BUCKET` | `str` | `synth-city` | S3 bucket name |

If these credentials are not set, the pipeline still works — it silently skips remote storage. All results remain local in `WORKSPACE_DIR`.

### Storage Layout

```
synth-city/                                  (bucket)
├── experiments/{run_id}/{name}.json         individual experiment result
├── pipeline_runs/{run_id}/summary.json      full pipeline run summary
├── pipeline_runs/{run_id}/comparison.json   CRPS ranking
├── pipeline_runs/latest.json                pointer to most recent run
└── scores/                                  scoring emulator data
    ├── prompts/{YYYY-MM-DD}/{prompt_id}.json
    ├── daily/{YYYY-MM-DD}.json
    └── leaderboard.json
```

---

## Pipeline Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_AGENT_TURNS` | `int` | `50` | Safety cap: max LLM round-trips per agent |
| `WORKSPACE_DIR` | `Path` | `./workspace` | Working directory for experiment artifacts |

---

## Logging

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LOG_DIR` | `Path` | `./logs` | Log output directory |
| `LOG_LEVEL` | `str` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

Logging uses `RotatingFileHandler` with 20 MB file size and 5 backup files at DEBUG level.

---

## OpenClaw Bridge Server

Configuration for the HTTP bridge that OpenClaw bots connect to.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BRIDGE_HOST` | `str` | `127.0.0.1` | Server bind address |
| `BRIDGE_PORT` | `int` | `8377` | Server port |
| `BRIDGE_API_KEY` | `str` | — | Optional API key for authentication |

When `BRIDGE_API_KEY` is set, all requests must include a matching `X-API-Key` header. HMAC-safe comparison is used.

---

## Multi-Bot Concurrency

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BOT_SESSION_TTL_SECONDS` | `int` | `3600` | Idle session timeout (seconds) |
| `MAX_CONCURRENT_PIPELINES` | `int` | `10` | Maximum parallel pipeline runs across all bots |

The session registry runs a background reaper thread that evicts idle sessions after `BOT_SESSION_TTL_SECONDS`. The "default" session, sessions with active pipelines, and sessions with in-flight HTTP requests are never evicted.

---

## Example `.env` File

```bash
# === LLM Inference (Required) ===
CHUTES_API_KEY=your-chutes-api-key
CHUTES_BASE_URL=https://llm.chutes.ai/v1
DEFAULT_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct

# === Per-Agent Models (Optional) ===
# PLANNER_MODEL=Qwen/Qwen3-235B-A22B
# TRAINER_MODEL=Qwen/Qwen3-235B-A22B
# AUTHOR_MODEL=Qwen/Qwen3-235B-A22B

# === GPU Compute (Optional) ===
# BASILICA_API_TOKEN=your-basilica-token
# BASILICA_MAX_HOURLY_RATE=0.44
# BASILICA_ALLOWED_GPU_TYPES=TESLA V100,RTX-A4000,RTX-A6000

# === Bittensor (Optional) ===
# BT_WALLET_NAME=default
# BT_HOTKEY_NAME=default
# BT_NETWORK=finney

# === Storage (Optional) ===
# HIPPIUS_ENDPOINT=https://s3.hippius.network
# HIPPIUS_ACCESS_KEY=your-access-key
# HIPPIUS_SECRET_KEY=your-secret-key

# === Publishing (Optional) ===
# HF_TOKEN=your-hf-token
# HF_REPO_ID=your-username/your-model-repo
# WANDB_PROJECT=synth-city

# === Research Defaults ===
# RESEARCH_N_PATHS=100
# RESEARCH_D_MODEL=32
# RESEARCH_EPOCHS=1

# === Bridge Server ===
# BRIDGE_HOST=127.0.0.1
# BRIDGE_PORT=8377
# BRIDGE_API_KEY=your-secret-key
```
