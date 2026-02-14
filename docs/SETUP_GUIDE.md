# synth-city Setup Guide

Complete guide for installing, configuring, and running synth-city — an autonomous AI research pipeline that discovers, trains, debugs, and publishes probabilistic price forecasting models for **Bittensor Subnet 50 (Synth)**.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the Pipeline](#running-the-pipeline)
6. [CLI Commands Reference](#cli-commands-reference)
7. [HTTP Bridge Server](#http-bridge-server)
8. [Experiment History](#experiment-history)
9. [Bittensor Mining](#bittensor-mining)
10. [Basilica GPU Compute](#basilica-gpu-compute)
11. [Hippius Decentralised Storage](#hippius-decentralised-storage)
12. [Agent Architecture](#agent-architecture)
13. [How Agents Run Experiments](#how-agents-run-experiments)
14. [Tool System](#tool-system)
15. [SN50 Competition Details](#sn50-competition-details)
16. [Development Workflow](#development-workflow)
17. [Extending synth-city](#extending-synth-city)
18. [Project Structure](#project-structure)
19. [Troubleshooting](#troubleshooting)

---

## Overview

synth-city wraps [open-synth-miner](https://github.com/tensorlink-dev/open-synth-miner) (a composable PyTorch framework for probabilistic forecasting) and chains specialized AI agents together to automate the entire research cycle:

```
Planner → Trainer → CodeChecker → Debugger → Publisher
```

Each agent has access to scoped tools and carefully crafted prompts. The pipeline discovers available model components, designs experiments, trains and evaluates them, validates results against SN50 specifications, fixes failures, and optionally publishes winning models to Hugging Face Hub and Weights & Biases.

---

## Prerequisites

### Required

| Requirement | Notes |
|-------------|-------|
| **Python 3.10+** | Tested on 3.10, 3.11, 3.12 |
| **Git** | For cloning repos |
| **Chutes AI API key** | LLM inference provider (OpenAI-compatible). Sign up at [chutes.ai](https://chutes.ai/) |
| **~4 GB disk space** | For PyTorch, dependencies, and workspace artifacts |

### Optional (depending on features used)

| Requirement | Needed for |
|-------------|------------|
| **CUDA-capable GPU** | Faster local training (CPU works but is slower) |
| **Basilica API key** | Decentralised GPU training via Bittensor SN39 |
| **Hugging Face token** | Publishing models to HF Hub |
| **Weights & Biases account** | Experiment tracking and trend analysis |
| **Bittensor wallet** | Submitting predictions as a miner on SN50 |
| **Hippius / S3 credentials** | Persistent experiment storage |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/tensorlink-dev/synth-city.git
cd synth-city
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

**Option A — from requirements.txt (recommended):**

```bash
pip install -r requirements.txt
```

**Option B — editable install with dev tools:**

```bash
pip install -e ".[dev]"
```

This installs everything from Option A plus pytest, ruff, and mypy for development.

### 4. Install open-synth-miner

synth-city's research tools are built on top of `open-synth-miner`, which provides the composable backbone blocks, head types, presets, and the `ResearchSession` API.

**Option A — editable install from a local clone (recommended for development):**

```bash
git clone https://github.com/tensorlink-dev/open-synth-miner.git ../open-synth-miner
pip install -e ../open-synth-miner
```

**Option B — install directly from GitHub:**

```bash
pip install "open-synth-miner @ git+https://github.com/tensorlink-dev/open-synth-miner.git"
```

### 5. Verify the installation

```bash
python -c "from research import ResearchSession; print('open-synth-miner OK')"
python -c "from pipeline.orchestrator import PipelineOrchestrator; print('synth-city OK')"
```

Both commands should print their respective "OK" messages without errors.

---

## Configuration

All configuration is managed through environment variables, loaded from a `.env` file via [python-dotenv](https://pypi.org/project/python-dotenv/).

### Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` in your editor and fill in the values below. Only the Chutes AI section is strictly required to run the pipeline.

### LLM Inference (Required)

```env
CHUTES_API_KEY=your_chutes_api_key_here
CHUTES_BASE_URL=https://llm.chutes.ai/v1
DEFAULT_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
```

The `CHUTES_BASE_URL` points to the Chutes AI endpoint, which exposes an OpenAI-compatible chat completions API. Any provider that speaks the same protocol can be substituted (e.g., OpenRouter, vLLM, Ollama, Together AI). Just change the base URL and API key.

### Per-Agent Model Overrides (Optional)

Each agent can use a different LLM. Larger models are used where reasoning quality matters most (Planner, Trainer, Author); smaller code-tuned models handle structured/mechanical tasks (CodeChecker, Debugger, Publisher).

```env
PLANNER_MODEL=Qwen/Qwen3-235B-A22B
TRAINER_MODEL=Qwen/Qwen3-235B-A22B
AUTHOR_MODEL=Qwen/Qwen3-235B-A22B
CODECHECKER_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
DEBUGGER_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
PUBLISHER_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
```

If an agent-specific model is not set, it falls back to `DEFAULT_MODEL`. The model selection logic is in `config.py:model_for()`.

### Research Defaults (Optional)

These control the default hyperparameters for experiments created through the tool system:

```env
RESEARCH_N_PATHS=100        # Monte Carlo paths generated per experiment
RESEARCH_D_MODEL=32         # Hidden dimension of backbone blocks
RESEARCH_HORIZON=12         # Prediction steps (at 5-min intervals)
RESEARCH_SEQ_LEN=32         # Input sequence length fed to the model
RESEARCH_FEATURE_DIM=4      # Number of input features (OHLC)
RESEARCH_BATCH_SIZE=4       # Training batch size
RESEARCH_LR=0.001           # Learning rate
RESEARCH_EPOCHS=1           # Training epochs per experiment
```

These are deliberately small defaults for fast iteration during research. For production-quality results, increase `RESEARCH_N_PATHS` to 1000, `RESEARCH_EPOCHS` to 5+, and `RESEARCH_D_MODEL` to 64 or 128.

### Basilica GPU Compute (Optional)

For offloading training to decentralised GPUs on Bittensor SN39 (see [Basilica GPU Compute](#basilica-gpu-compute)):

```env
BASILICA_API_KEY=your_basilica_api_key_here
BASILICA_ENDPOINT=https://api.basilica.tplr.ai
```

### Bittensor Wallet (Optional)

For submitting predictions as a miner on SN50 (see [Bittensor Mining](#bittensor-mining)):

```env
BT_WALLET_NAME=default
BT_HOTKEY_NAME=default
BT_NETWORK=finney
BT_NETUID=50
```

### Market Data

```env
PYTH_PRICE_FEED_URL=https://hermes.pyth.network
```

Price data is fetched from the [Pyth Network](https://pyth.network/) oracle. The default endpoint works without authentication.

### Publishing (Optional)

```env
HF_REPO_ID=your-username/SN50-Hybrid-Hub
WANDB_PROJECT=synth-city
```

Before publishing, authenticate with each service:

```bash
# Hugging Face
pip install huggingface-cli
huggingface-cli login

# Weights & Biases
pip install wandb
wandb login
```

### Hippius Decentralised Storage (Optional)

S3-compatible storage for persisting experiment results across runs (see [Hippius Decentralised Storage](#hippius-decentralised-storage)):

```env
HIPPIUS_ENDPOINT=https://s3.hippius.network
HIPPIUS_ACCESS_KEY=your_hippius_access_key
HIPPIUS_SECRET_KEY=your_hippius_secret_key
HIPPIUS_BUCKET=synth-city
```

Any S3-compatible endpoint works (Cloudflare R2, MinIO, AWS S3).

### Pipeline Settings

```env
MAX_AGENT_TURNS=50          # Safety cap: max LLM round-trips per agent
WORKSPACE_DIR=./workspace   # Where experiment artifacts are saved
```

### OpenClaw Bridge Server

```env
BRIDGE_HOST=127.0.0.1
BRIDGE_PORT=8377
```

---

## Running the Pipeline

### Full Agentic Pipeline

This is the primary way to use synth-city. It runs all agents in sequence:

```bash
python main.py pipeline
```

What happens:

1. **Planner** discovers available blocks/heads/presets and produces an experiment plan.
2. **Trainer** executes the plan — creates, validates, and runs experiments, tracking metrics.
3. **CodeChecker** validates the best experiment config against SN50 specifications.
4. If validation fails, **Debugger** fixes the config and re-runs. This check/debug loop repeats until validation passes or retries are exhausted.
5. **(Optional)** **Publisher** pushes the best model to Hugging Face Hub and logs to W&B.

To include the publish step:

```bash
python main.py pipeline --publish
```

Additional flags:

```bash
python main.py pipeline \
  --channel default \
  --retries 5 \
  --temperature 0.1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--channel` | `default` | Prompt fragment channel (for A/B testing prompts) |
| `--retries` | `5` | Max retry attempts per stage |
| `--temperature` | `0.1` | Starting LLM temperature (escalates on failure) |
| `--publish` | off | Publish best model to HF Hub + W&B |

### Quick Baseline Sweep (No Agents)

Bypass the agents entirely and run experiments directly through `ResearchSession`:

```bash
# Run all 10 presets
python main.py sweep

# Run specific presets
python main.py sweep --presets transformer_lstm,pure_transformer,conv_gru

# Run with more epochs
python main.py sweep --epochs 3
```

### Single Experiment

Run one specific model configuration:

```bash
python main.py experiment \
  --blocks TransformerBlock,LSTMBlock \
  --head SDEHead \
  --d-model 64 \
  --horizon 12 \
  --n-paths 100 \
  --lr 0.001 \
  --epochs 1
```

### Quick One-Liner

Shorthand for running a single experiment with sane defaults:

```bash
python main.py quick --blocks TransformerBlock,LSTMBlock
python main.py quick --blocks TransformerBlock --head GBMHead --d-model 32
```

### Run a Single Agent (Debugging)

Useful for testing individual agent behavior:

```bash
python main.py agent --name planner
python main.py agent --name trainer
python main.py agent --name codechecker
python main.py agent --name debugger --message "Fix this config: {...}"
python main.py agent --name publisher
python main.py agent --name author --temperature 0.3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | (required) | Agent name: planner, codechecker, debugger, trainer, publisher, author |
| `--message` | `"Begin the task."` | Custom user message to send to the agent |
| `--temperature` | `0.1` | LLM temperature |

---

## CLI Commands Reference

| Command | Description |
|---------|-------------|
| `python main.py pipeline` | Full agentic pipeline (Planner → Trainer → Checker/Debugger → Publisher) |
| `python main.py sweep` | Direct preset sweep without agents |
| `python main.py experiment` | Run a single experiment with explicit config |
| `python main.py quick` | One-liner experiment with defaults |
| `python main.py bridge` | Start the HTTP bridge server |
| `python main.py client` | CLI client to interact with the bridge server |
| `python main.py history` | Query experiment history from Hippius, W&B, or HF Hub |
| `python main.py agent` | Run a single agent in isolation |

---

## HTTP Bridge Server

The bridge server exposes synth-city's pipeline operations and research tools as a lightweight HTTP API. It works standalone or as a backend for agent frameworks like OpenClaw.

### Starting the server

```bash
# Default: 127.0.0.1:8377
python main.py bridge

# Custom host/port
python main.py bridge --host 0.0.0.0 --port 9000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| POST | `/pipeline/run` | Start a full pipeline run (async, runs in background) |
| GET | `/pipeline/status` | Poll current pipeline status |
| POST | `/experiment/create` | Create an experiment config |
| POST | `/experiment/run` | Run an experiment and return metrics |
| POST | `/experiment/validate` | Validate an experiment config |
| GET | `/experiment/compare` | Compare all session results |
| GET | `/components/blocks` | List available backbone blocks |
| GET | `/components/heads` | List available head types |
| GET | `/components/presets` | List ready-to-run presets |
| GET | `/session/summary` | Current research session summary |
| POST | `/session/clear` | Reset research session |
| GET | `/market/price/:asset` | Live price from Pyth oracle |
| GET | `/market/history/:asset` | Historical OHLCV data |

### Example usage with curl

```bash
# Health check
curl http://127.0.0.1:8377/health

# List available blocks
curl http://127.0.0.1:8377/components/blocks

# Create an experiment
curl -X POST http://127.0.0.1:8377/experiment/create \
  -H "Content-Type: application/json" \
  -d '{"blocks": ["TransformerBlock", "LSTMBlock"], "head": "SDEHead", "d_model": 64}'

# Kick off a pipeline run
curl -X POST http://127.0.0.1:8377/pipeline/run -d '{}'

# Check pipeline status
curl http://127.0.0.1:8377/pipeline/status

# Get BTC price
curl http://127.0.0.1:8377/market/price/BTC
```

### Using the built-in CLI client

You do not need curl — a built-in CLI client talks to the bridge:

```bash
# Start the bridge in one terminal
python main.py bridge

# In another terminal
python main.py client health
python main.py client blocks
python main.py client heads
python main.py client presets
python main.py client price BTC
python main.py client history BTC 30
python main.py client run                  # trigger a full pipeline run
python main.py client run --publish        # trigger pipeline run + publish
python main.py client status               # check pipeline status
python main.py client compare              # compare experiment results
python main.py client summary              # session summary
python main.py client clear                # reset session
```

---

## Experiment History

synth-city can query experiment history from three backends: Hippius (decentralised storage), Weights & Biases, and Hugging Face Hub.

### Hippius history

```bash
# List recent experiments (sorted by CRPS, best first)
python main.py history hippius

# Limit results
python main.py history hippius --limit 10

# Load a specific run
python main.py history hippius --run-id 20250101-120000-abc12345

# Load the most recent run
python main.py history hippius --run-id latest
```

### Weights & Biases history

```bash
# List runs sorted by best CRPS
python main.py history wandb --order best

# List most recent runs
python main.py history wandb --order recent --limit 10

# Show CRPS trends over time
python main.py history wandb --trends
```

### Hugging Face Hub history

```bash
# List models in the default repo
python main.py history hf

# Specify a different repo
python main.py history hf --repo-id your-username/your-repo
```

---

## Bittensor Mining

To compete as a miner on Subnet 50, you need a Bittensor wallet and registered hotkey.

### 1. Install Bittensor

```bash
pip install bittensor
```

### 2. Create a wallet

```bash
btcli wallet new_coldkey --wallet.name default
btcli wallet new_hotkey --wallet.name default --wallet.hotkey default
```

### 3. Register on SN50

Registration requires TAO. Check the current registration cost with:

```bash
btcli subnet register --netuid 50 --wallet.name default --wallet.hotkey default
```

### 4. Configure synth-city

Set the following in your `.env`:

```env
BT_WALLET_NAME=default
BT_HOTKEY_NAME=default
BT_NETWORK=finney
BT_NETUID=50
```

### 5. Train and deploy a model

Use the pipeline to find the best model, then run it as a miner. The `SynthMiner` class in `subnet/miner.py` handles:

- Receiving prediction requests from validators
- Fetching latest prices for each asset
- Generating 1,000 Monte Carlo paths using your best model
- Formatting and returning the prediction payload

### Prediction requirements

Each prediction must produce:

- **1,000 Monte Carlo price paths** per asset
- **289 timesteps** (24 hours at 5-minute intervals, including t0)
- **13 timesteps** for HFT (1 hour at 5-minute intervals)
- Paths must represent continuous probability distributions, not point forecasts
- All values must be positive (prices), contain no NaN or Inf values

---

## Basilica GPU Compute

Basilica provides decentralised GPU compute via Bittensor SN39. Use it to offload training to remote A100 GPUs.

### Setup

1. Get a Basilica API key from [basilica.tplr.ai](https://api.basilica.tplr.ai)
2. Add to your `.env`:

```env
BASILICA_API_KEY=your_basilica_api_key_here
BASILICA_ENDPOINT=https://api.basilica.tplr.ai
```

### How it works

The `BasilicaClient` (`compute/basilica.py`) submits containerised training jobs:

```python
from compute.basilica import BasilicaClient, JobSpec

client = BasilicaClient()

# Submit a training job
job_id = client.submit(JobSpec(
    script_path="train.py",
    gpu_type="A100",         # GPU type
    num_gpus=1,              # Number of GPUs
    timeout_minutes=60,      # Max runtime
))

# Poll until completion
status = client.poll(job_id, timeout=3600)

if status.status == "completed":
    # Download trained model artifacts
    client.download_artifacts(job_id, "output/")
elif status.status == "failed":
    print(f"Job failed: {status.error}")
```

### Job defaults

| Setting | Default |
|---------|---------|
| GPU type | A100 |
| Number of GPUs | 1 |
| Docker image | `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` |
| Timeout | 60 minutes |

---

## Hippius Decentralised Storage

Hippius provides S3-compatible decentralised storage on the Bittensor network. synth-city uses it to persist experiment results across pipeline runs so agents can learn from past experiments.

### Setup

Any S3-compatible endpoint works — Hippius, Cloudflare R2, MinIO, or AWS S3.

```env
HIPPIUS_ENDPOINT=https://s3.hippius.network
HIPPIUS_ACCESS_KEY=your_access_key
HIPPIUS_SECRET_KEY=your_secret_key
HIPPIUS_BUCKET=synth-city
```

If these credentials are not set, the pipeline still works — it just skips remote storage (all results remain local in `WORKSPACE_DIR`).

### Storage layout

```
synth-city/                          (bucket)
  experiments/{run_id}/{name}.json   individual experiment result
  pipeline_runs/{run_id}/summary.json    full pipeline run summary
  pipeline_runs/{run_id}/comparison.json CRPS ranking at end of run
  pipeline_runs/latest.json              pointer to most recent run
```

### Agent tools

Agents can interact with Hippius via registered tools:

- `save_to_hippius` — Save an experiment result
- `list_hippius_runs` — List all pipeline runs
- `load_hippius_run` — Load a specific run (or `latest`)
- `load_hippius_history` — Load all experiments ranked by CRPS

---

## Agent Architecture

### Core: SimpleAgent

All agents are powered by a single ~200-line `SimpleAgent` class (`pipeline/providers/simple_agent.py`) that implements a straightforward loop:

```
1. Send messages (system prompt + conversation history) to LLM
2. If LLM responds with tool calls → execute them
3. Append tool results to conversation
4. Repeat until the agent calls finish() or hits MAX_AGENT_TURNS
```

There are no DAGs, no complex state machines — just a sequential message-passing loop. The sophistication lives in the **prompts** and **orchestration**, not in the agent framework itself.

### Specialization via Composition

Each agent is a thin wrapper (`BaseAgentWrapper` subclass) that configures:

- **System prompt** — assembled from composable prompt fragments in `pipeline/prompts/`
- **Tool set** — which tools from the registry the agent can access
- **Context** — prior results or error reports injected as user messages

### The 6 Agents

#### 1. Planner

**Purpose:** Discover what building blocks are available and design an experiment plan.

**Tools:** `list_blocks`, `list_heads`, `list_presets`, `session_summary`, `compare_results`

**Behavior — Two-Phase Reasoning:**

- **Phase 1 (DIAGNOSTIC):** Call discovery tools to enumerate the 15 backbone blocks, 6 head types, and 10 presets. Review any prior experiment results.
- **Phase 2 (EXECUTION PLAN):** Based on the diagnostic, produce a structured JSON plan specifying which block/head/d_model combinations to test, with rationale and hyperparameter ranges.

**Output:** A JSON plan consumed by the Trainer.

#### 2. Trainer

**Purpose:** Execute the Planner's experiment plan and find the best configuration.

**Tools:** `create_experiment`, `validate_experiment`, `run_experiment`, `run_preset`, `sweep_presets`, `compare_results`, `session_summary`

**Behavior:**

1. Receives the Planner's plan as context
2. Creates experiments using `create_experiment(blocks, head, d_model, ...)`
3. Validates each experiment config before running
4. Runs experiments and collects metrics (CRPS, sharpness, log-likelihood)
5. Uses `compare_results()` to rank experiments
6. Reports the best experiment and its metrics

**Output:** Best experiment config + metrics dictionary.

#### 3. CodeChecker

**Purpose:** Validate that the best experiment config conforms to SN50 specifications.

**Tools:** `validate_experiment`, `describe_experiment`, `list_blocks`, `list_heads`

**Behavior:**

1. Receives the best experiment config from the Trainer
2. **Must** call `validate_experiment()` before finishing (mandatory tool use)
3. Checks tensor shapes, block compatibility, head requirements, parameter counts
4. Returns PASS or FAIL with specific validation errors

**Output:** PASS/FAIL verdict with error details.

#### 4. Debugger

**Purpose:** Fix experiments that failed validation or execution.

**Tools:** `create_experiment`, `validate_experiment`, `run_experiment`, `list_blocks`, `list_heads`

**Behavior:**

1. Receives the error report + failed config as context
2. Analyzes the failure (shape mismatch, invalid block combo, config error, etc.)
3. Creates a modified experiment with fixes applied
4. Validates and re-runs the fixed experiment
5. **Stall detection:** If the Debugger produces the same config as the previous attempt, the orchestrator injects a `CRITICAL WARNING` forcing it to try a fundamentally different approach

**Output:** Fixed experiment config + successful run metrics.

#### 5. Publisher

**Purpose:** Push the validated model to production tracking systems.

**Tools:** `validate_experiment`, `publish_model`, `log_to_wandb`

**Behavior:**

1. Receives the best validated experiment + metrics
2. Performs a final validation check
3. Publishes the model config to Hugging Face Hub
4. Logs the experiment run to Weights & Biases
5. Returns links to both

**Output:** HF Hub URL + W&B run URL.

#### 6. Author

**Purpose:** Write new backbone blocks or prediction heads into the open-synth-miner registry.

**Tools:** Component registration tools, file I/O

**Behavior:**

1. Receives a description of the desired component
2. Generates PyTorch code following the uniform tensor interface
3. Registers the component with the open-synth-miner registry

**Output:** New component code + registry update.

---

## How Agents Run Experiments

### End-to-End Data Flow

Here is exactly what happens when you run `python main.py pipeline --publish`:

```
┌─────────────────────────────────────────────────────────┐
│                    PipelineOrchestrator                  │
│                                                         │
│  ┌─────────┐    ┌─────────┐    ┌────────┐   ┌────────┐ │
│  │ Planner │───>│ Trainer │───>│ Check/ │──>│Publish │ │
│  │         │    │         │    │ Debug  │   │        │ │
│  └─────────┘    └─────────┘    │ Loop   │   └────────┘ │
│                                └────────┘               │
│                                                         │
│  Retry loops + temperature escalation at every stage    │
└─────────────────────────────────────────────────────────┘
```

**Stage 1 — Planning:**

```
Planner agent starts (temp=0.1)
  → list_blocks()       → discovers 15 blocks (TransformerBlock, LSTMBlock, GRUBlock, ...)
  → list_heads()        → discovers 6 heads (GBMHead, GARCHHead, SDEHead, ...)
  → list_presets()      → discovers 10 presets (transformer_lstm, pure_transformer, ...)
  → session_summary()   → reviews any prior experiment results
  → finish(plan_json)   → structured plan of experiments to run
```

**Stage 2 — Training:**

```
Trainer agent starts (temp=0.1), receives plan as context
  For each experiment in the plan:
    → create_experiment(blocks=[...], head="...", d_model=32)
    → validate_experiment(experiment_id)
    → run_experiment(experiment_id)  →  returns {crps, sharpness, log_likelihood}
  → compare_results()               →  ranks all experiments
  → finish(best_experiment, best_metrics)
```

**Stage 3 — Validation & Debugging (loop):**

```
CodeChecker receives best experiment
  → validate_experiment()
  → describe_experiment()
  → finish(PASS) or finish(FAIL, errors=[...])

If FAIL:
  Debugger receives error report + failed config
    → create_experiment(modified_config)
    → validate_experiment()
    → run_experiment()
    → finish(fixed_config, new_metrics)

  Back to CodeChecker...
  (repeats until PASS or max retries exhausted)
```

**Stage 4 — Publishing (optional):**

```
Publisher receives validated experiment + metrics
  → validate_experiment()       (final check)
  → publish_model(experiment)   → pushes to HF Hub
  → log_to_wandb(metrics)       → logs to W&B
  → finish(hf_url, wandb_url)
```

### Experiment Format

When agents create experiments, the config looks like this:

```json
{
  "model": {
    "backbone": {
      "blocks": ["TransformerBlock", "LSTMBlock"],
      "d_model": 32,
      "feature_dim": 4,
      "seq_len": 32
    },
    "head": {
      "_target_": "...NeuralSDEHead",
      "latent_size": 32
    }
  },
  "training": {
    "horizon": 12,
    "n_paths": 100,
    "batch_size": 4,
    "lr": 0.001,
    "epochs": 1
  }
}
```

And the run result:

```json
{
  "status": "ok",
  "metrics": {
    "crps": 0.0123,
    "sharpness": 15.45,
    "log_likelihood": -2.34
  },
  "param_count": 45320,
  "training_time_s": 123.45
}
```

### Resilience Mechanisms

The orchestrator employs several strategies to handle failures:

- **Temperature escalation:** Starting at 0.1, the LLM temperature increases by a configurable step (default 0.1) on each retry — encouraging more creative solutions after repeated failures.
- **Stall detection:** If the Debugger produces an identical experiment config to its previous attempt, the orchestrator injects a `CRITICAL WARNING` into the conversation forcing the agent to try a fundamentally different approach.
- **Retry loops:** Each stage has a configurable max retries (default 5). The pipeline moves to the next stage only on success.
- **Ephemeral compression:** Large tool outputs and context blobs are truncated before being passed to downstream agents, preventing token overflow on long runs.

---

## Tool System

Tools are registered globally via a decorator-based registry (`pipeline/tools/registry.py`):

```python
from pipeline.tools.registry import tool

@tool()
def list_blocks() -> dict:
    """List all available backbone blocks."""
    ...
```

The registry:
- Auto-generates OpenAI-compatible JSON schemas from type hints
- Provides per-agent tool scoping (agents request tools by name via `build_toolset()`)
- Handles argument coercion for sloppy LLM outputs (empty strings → empty lists, JSON-in-strings, string booleans)

### Tool Groups

| Category | Tools | Used By |
|----------|-------|---------|
| Discovery | `list_blocks`, `list_heads`, `list_presets` | Planner, Debugger |
| Experiment CRUD | `create_experiment`, `validate_experiment`, `describe_experiment` | Trainer, Checker, Debugger |
| Execution | `run_experiment`, `run_preset`, `sweep_presets` | Trainer, Debugger |
| Analysis | `compare_results`, `session_summary` | Planner, Trainer |
| Publishing | `publish_model`, `log_to_wandb` | Publisher |
| Training | `run_training_local`, `submit_basilica_job` | (advanced) |
| Storage | `save_to_hippius`, `list_hippius_runs`, `load_hippius_run`, `load_hippius_history` | (advanced) |
| Utilities | `run_python`, `read_file`, `write_file` | (advanced) |
| Market Data | `get_price`, `get_price_history` | (advanced) |

---

## SN50 Competition Details

### What is Synth (SN50)?

Synth is Bittensor Subnet 50 — a decentralised probabilistic price forecasting network. Miners compete to produce the most well-calibrated price distribution forecasts across 9 assets.

### Prediction Requirements

Each miner must produce:
- **1,000 Monte Carlo price paths** per asset
- **289 timesteps** (24 hours at 5-minute intervals, including t0)
- Paths must represent **continuous probability distributions**, not point forecasts

### Tracked Assets and Scoring Weights

| Asset | Description | Weight |
|-------|-------------|--------|
| BTC | Bitcoin | 1.00 |
| ETH | Ethereum | 0.67 |
| SOL | Solana | 0.59 |
| XAU | Gold | 2.26 |
| SPYX | S&P 500 | 2.99 |
| NVDAX | NVIDIA | 1.39 |
| TSLAX | Tesla | 1.42 |
| AAPLX | Apple | 1.86 |
| GOOGLX | Alphabet | 1.43 |

Higher weights mean the asset contributes more to your overall score. SPYX and XAU are weighted most heavily.

### Scoring: CRPS

Predictions are scored using **CRPS** (Continuous Ranked Probability Score):

```
CRPS = (1/N) * sum(|y_n - x|) - (1/2N²) * sum(sum(|y_n - y_m|))
```

- **Lower CRPS = better** calibrated distribution
- Measured at increments: 5, 10, 15, 30, 60, 180, 360, 720, 1440 minutes
- Per-asset scores are weighted by the asset weights above
- CRPS rewards distributions that are both **accurate** (centered on the true value) and **sharp** (tight, not overly dispersed)

### Leaderboard

- Rolling window: **10 days**
- Score cap percentile: **90%** (scores worse than the 90th percentile are capped)
- Emission allocation uses softmax with beta = **-0.1**

### Model Components (from open-synth-miner)

**15 Backbone Blocks:**
TransformerBlock, LSTMBlock, GRUBlock, ResConvBlock, FourierBlock, TimesNetBlock, WaveNetBlock, and more — each implementing a different approach to sequence modeling. All blocks share a uniform tensor interface: `(batch, seq, d_model) → (batch, seq, d_model)`.

**6 Head Types (increasing expressiveness):**

| Head | Description | Complexity |
|------|-------------|------------|
| GBMHead | Geometric Brownian Motion | Simplest baseline |
| GARCHHead | GARCH volatility clustering | Low |
| HestonHead | Heston stochastic volatility | Medium |
| SDEHead | Neural SDE (learned drift + diffusion) | Medium-High |
| FlowHead | Normalizing flow | High |
| NeuralSDEHead | Full neural SDE | Highest |

**10 Presets:**
Pre-configured block+head combinations ready to run: `transformer_lstm`, `pure_transformer`, `conv_gru`, `timesnet_fourier`, etc.

---

## Development Workflow

### Linting

synth-city uses [ruff](https://docs.astral.sh/ruff/) for linting (rules: E, F, W, I; line length: 100).

```bash
# Check for lint errors
ruff check .

# Auto-fix what can be fixed
ruff check --fix .
```

### Type Checking

```bash
mypy .
```

### Testing

```bash
# Run the full test suite
pytest

# Run a specific test
pytest -k test_name

# Run with verbose output
pytest -v
```

Tests are configured in `pyproject.toml` with `asyncio_mode = "auto"` and test discovery in the `tests/` directory.

### Code Conventions

- Python 3.10+, `from __future__ import annotations` in every file
- Type hints on all function signatures using modern syntax (`dict[str, Any]` not `Dict`)
- Ruff: line length 100, rules E/F/W/I
- Composition over inheritance
- Environment-based config via `python-dotenv`

---

## Extending synth-city

### Adding a New Tool

1. Create a function with the `@tool` decorator in `pipeline/tools/`:

```python
# pipeline/tools/my_tools.py
from pipeline.tools.registry import tool

@tool(description="Short description of what this tool does.")
def my_new_tool(param1: str, param2: int = 10) -> str:
    """Detailed docstring."""
    # Implementation
    return json.dumps({"result": "..."})
```

2. Add the tool name to the relevant agent's `build_tools()` method in `pipeline/agents/`.

3. Import the module somewhere so the decorator runs (e.g., in the agent's module or in a central init).

### Adding a New Agent

1. Create a subclass of `BaseAgentWrapper` in `pipeline/agents/`:

```python
# pipeline/agents/my_agent.py
from pipeline.agents.base import BaseAgentWrapper

class MyAgent(BaseAgentWrapper):
    agent_name = "myagent"

    def build_system_prompt(self) -> str:
        return "You are a specialized agent that..."

    def build_tools(self) -> list[str]:
        return ["tool_a", "tool_b", "tool_c"]
```

2. Add a prompt module in `pipeline/prompts/` if the agent needs complex prompt logic.

3. Register the agent in `main.py`'s `cmd_agent()` function to make it accessible from the CLI.

### Adding a New Backbone Block or Head

Use the ComponentAuthor agent:

```bash
python main.py agent --name author --message "Create a MambaBlock that uses selective state spaces"
```

Or manually add components to the `open-synth-miner` registry following the uniform tensor interface: `(batch, seq, d_model) → (batch, seq, d_model)`.

---

## Project Structure

```
synth-city/
├── main.py                          # CLI entry point (8 subcommands)
├── config.py                        # Environment-based config (all settings via .env)
├── pyproject.toml                   # Package metadata, dependencies, tool config
├── requirements.txt                 # Direct dependency listing
├── .env.example                     # Template environment file
│
├── pipeline/
│   ├── orchestrator.py              # Retry loops, temperature escalation, stall detection
│   ├── providers/
│   │   ├── simple_agent.py          # ~200-line core agent loop
│   │   └── chutes_client.py         # OpenAI-compatible LLM client
│   ├── agents/
│   │   ├── base.py                  # BaseAgentWrapper — composition pattern
│   │   ├── planner.py               # Discovers components, produces experiment plan
│   │   ├── trainer.py               # Executes experiments via ResearchSession
│   │   ├── code_checker.py          # Validates experiment configs + results
│   │   ├── debugger.py              # Fixes failed experiments
│   │   ├── publisher.py             # HF Hub + W&B production tracking
│   │   └── author.py                # Writes new blocks/heads into registry
│   ├── tools/
│   │   ├── registry.py              # @tool decorator, global registry, build_toolset()
│   │   ├── research_tools.py        # ResearchSession API (create/run/validate/compare)
│   │   ├── publish_tools.py         # HF Hub + W&B logging
│   │   ├── file_tools.py            # write_file, read_file
│   │   ├── check_shapes.py          # SN50 shape validation
│   │   ├── market_data.py           # Price data fetching from Pyth
│   │   ├── training_tools.py        # Local + Basilica training
│   │   ├── register_tools.py        # Write components + reload registry
│   │   ├── analysis_tools.py        # W&B + HF Hub analysis
│   │   └── hippius_store.py         # S3-compatible decentralised storage
│   └── prompts/
│       ├── fragments.py             # Composable prompt building blocks
│       ├── planner_prompts.py       # Phased reasoning + component reference
│       ├── checker_prompts.py       # Validation checklist
│       ├── debugger_prompts.py      # Error pattern catalog
│       ├── trainer_prompts.py       # Experiment execution
│       ├── publisher_prompts.py     # Publishing procedure
│       └── author_prompts.py        # Component authoring guidelines
│
├── models/                          # Standalone fallback models (GBM, GARCH, Heston)
├── data/                            # Market data fetching + preprocessing
├── compute/
│   └── basilica.py                  # Decentralised GPU client (Bittensor SN39)
├── subnet/
│   ├── config.py                    # SN50 constants (timesteps, scoring increments)
│   ├── miner.py                     # Prediction generation + submission
│   └── validator.py                 # CRPS scoring logic
├── integrations/
│   └── openclaw/
│       ├── bridge.py                # HTTP bridge server
│       ├── client.py                # CLI client for the bridge
│       └── skill/                   # OpenClaw skill definition
├── tests/                           # Test suite (pytest)
├── docs/
│   └── SETUP_GUIDE.md              # This file
└── workspace/                       # Experiment artifacts (auto-created)
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'research'"

`open-synth-miner` is not installed. Follow the installation steps in [Installation > Step 4](#4-install-open-synth-miner).

### "CHUTES_API_KEY not set" or LLM API errors

- Verify `CHUTES_API_KEY` is set correctly in your `.env` file
- Check that `CHUTES_BASE_URL` is reachable: `curl https://llm.chutes.ai/v1/models`
- The default models (Qwen/Qwen3-235B-A22B, Qwen/Qwen2.5-Coder-32B-Instruct) must be available on your provider
- If using a different LLM provider, ensure it speaks the OpenAI chat completions protocol

### Agent gets stuck in a loop

The pipeline has built-in stall detection. If the same config is produced twice in a row, a warning is injected. If the problem persists:

- Increase `--retries` (default 5)
- Start with a higher `--temperature` (e.g., 0.3)
- Run `python main.py sweep` first to establish baseline results the Planner can reference
- Check the agent logs for repeated tool calls with identical arguments

### CRPS scores are poor

- Increase `RESEARCH_N_PATHS` to 1000 for proper Monte Carlo sampling
- Increase `RESEARCH_EPOCHS` for more training (5+ epochs recommended for production)
- Try a more expressive head (SDEHead or NeuralSDEHead instead of GBMHead)
- Increase `RESEARCH_D_MODEL` (64 or 128) for more model capacity
- Run a sweep first to find the best preset as a starting point: `python main.py sweep --epochs 3`

### Publishing fails

- Ensure `HF_REPO_ID` is set and you have write access to the Hugging Face repo
- Run `huggingface-cli login` to authenticate
- For W&B, run `wandb login` first
- Check that the `wandb` and `huggingface_hub` packages are installed

### Bridge server won't start

- Check that the port is not already in use: `lsof -i :8377`
- Try a different port: `python main.py bridge --port 9000`
- Ensure all dependencies are installed (`pip install -r requirements.txt`)

### Basilica job failures

- Verify `BASILICA_API_KEY` is set in `.env`
- Check the job status for error messages: the `JobStatus.error` field contains details
- Ensure your training script is self-contained and can run inside the Docker container
- Default timeout is 60 minutes — increase `timeout_minutes` for longer training runs

### Hippius storage not working

- If credentials are not set, the pipeline silently skips remote storage — this is expected
- Verify your S3 endpoint is reachable
- Check that the bucket exists or that your credentials have permission to create it
- Any S3-compatible backend works — test with MinIO locally if needed

### Out of memory (OOM) during training

- Reduce `RESEARCH_BATCH_SIZE` (default 4, try 2 or 1)
- Reduce `RESEARCH_D_MODEL` (default 32)
- Reduce `RESEARCH_N_PATHS` (default 100)
- Use Basilica GPU compute for larger experiments
- If using CUDA, check available GPU memory with `nvidia-smi`
