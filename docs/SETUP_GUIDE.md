# synth-city Setup & Agent Guide

Complete guide for setting up synth-city and understanding how the agentic pipeline discovers, trains, validates, and publishes competitive models on **Bittensor Subnet 50 (Synth)**.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Pipeline](#running-the-pipeline)
5. [CLI Commands Reference](#cli-commands-reference)
6. [Agent Architecture](#agent-architecture)
7. [How Agents Run Experiments](#how-agents-run-experiments)
8. [Tool System](#tool-system)
9. [SN50 Competition Details](#sn50-competition-details)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python 3.10+**
- **Git**
- **A Chutes AI API key** — used for LLM inference (OpenAI-compatible). Sign up at [chutes.ai](https://chutes.ai/).
- **(Optional) Basilica API key** — for decentralised GPU training via Bittensor SN39.
- **(Optional) Hugging Face token** — for publishing models to HF Hub.
- **(Optional) Weights & Biases account** — for experiment tracking.
- **(Optional) Bittensor wallet** — for submitting predictions on SN50.

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

```bash
pip install -r requirements.txt
```

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

---

## Configuration

### Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` in your editor and fill in the required values. Below is a walkthrough of every section.

### LLM Inference (Required)

```env
CHUTES_API_KEY=your_chutes_api_key_here
CHUTES_BASE_URL=https://llm.chutes.ai/v1
DEFAULT_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
```

The `CHUTES_BASE_URL` points to the Chutes AI endpoint, which exposes an OpenAI-compatible API. Any provider that speaks the OpenAI chat completions protocol can be substituted here (e.g., OpenRouter, vLLM, Ollama).

### Per-Agent Model Overrides (Optional)

Each agent can use a different LLM. Larger models (Qwen3-235B) are used where reasoning quality matters most (Planner, Trainer); smaller models (Qwen2.5-Coder-32B) are used for structured/mechanical tasks (CodeChecker, Debugger, Publisher).

```env
PLANNER_MODEL=Qwen/Qwen3-235B-A22B
TRAINER_MODEL=Qwen/Qwen3-235B-A22B
CODECHECKER_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
DEBUGGER_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
PUBLISHER_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
```

If an agent-specific model is not set, it falls back to `DEFAULT_MODEL`.

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

These are deliberately small defaults for fast iteration during research. For production, increase `RESEARCH_N_PATHS` (to 1000), `RESEARCH_EPOCHS`, and `RESEARCH_D_MODEL`.

### Basilica GPU Compute (Optional)

For offloading training to decentralised GPUs on Bittensor SN39:

```env
BASILICA_API_KEY=your_basilica_api_key_here
BASILICA_ENDPOINT=https://api.basilica.tplr.ai
```

### Bittensor Wallet (Optional)

For submitting predictions as a miner on SN50:

```env
BT_WALLET_NAME=default
BT_HOTKEY_NAME=default
BT_NETWORK=finney
BT_NETUID=50
```

### Publishing (Optional)

```env
HF_REPO_ID=your-username/SN50-Hybrid-Hub
WANDB_PROJECT=synth-city
```

### Pipeline Settings

```env
MAX_AGENT_TURNS=50          # Safety cap: max LLM round-trips per agent
WORKSPACE_DIR=./workspace   # Where experiment artifacts are saved
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
  --channel default \       # Prompt fragment channel (for A/B testing prompts)
  --retries 5 \             # Max retry attempts per stage
  --temperature 0.1         # Starting LLM temperature (escalates on failure)
```

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
python main.py quick --blocks TransformerBlock,LSTMBlock --head GBMHead --d-model 32
```

### Run a Single Agent (Debugging)

Useful for testing individual agent behavior:

```bash
python main.py agent --name planner
python main.py agent --name trainer
python main.py agent --name codechecker
python main.py agent --name debugger --message "Fix this config: {...}"
python main.py agent --name publisher
```

---

## CLI Commands Reference

| Command | Description |
|---------|-------------|
| `python main.py pipeline` | Full agentic pipeline (Planner → Trainer → Checker/Debugger → Publisher) |
| `python main.py sweep` | Direct preset sweep without agents |
| `python main.py experiment` | Run a single experiment with explicit config |
| `python main.py quick` | One-liner experiment with defaults |
| `python main.py bridge` | Start the HTTP bridge server for OpenClaw integration |
| `python main.py client` | CLI client to interact with the bridge server |
| `python main.py agent` | Run a single agent in isolation |

---

## Agent Architecture

### Core: SimpleAgent

All agents are powered by a single ~200-line `SimpleAgent` class (`pipeline/providers/simple_agent.py`) that implements a straightforward loop:

```
1. Send messages (system prompt + conversation history) to LLM
2. If LLM responds with tool calls → execute them
3. Append tool results to conversation
4. Repeat until the agent calls `finish()` or hits MAX_AGENT_TURNS
```

There are no DAGs, no complex state machines — just a sequential message-passing loop. The sophistication lives in the **prompts** and **orchestration**, not in the agent framework itself.

### Specialization via Composition

Each of the 5 agents is a thin wrapper (`BaseAgentWrapper` subclass) that configures:

- **System prompt** — assembled from composable prompt fragments
- **Tool set** — which tools from the registry the agent can access
- **Context** — prior results or error reports injected as user messages

### The 5 Agents

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
- Provides per-agent tool scoping (agents request tools by name)
- Handles argument coercion for sloppy LLM outputs (empty strings → empty lists, JSON-in-strings, etc.)

### Tool Groups

| Category | Tools | Used By |
|----------|-------|---------|
| Discovery | `list_blocks`, `list_heads`, `list_presets` | Planner, Debugger |
| Experiment CRUD | `create_experiment`, `validate_experiment`, `describe_experiment` | Trainer, Checker, Debugger |
| Execution | `run_experiment`, `run_preset`, `sweep_presets` | Trainer, Debugger |
| Analysis | `compare_results`, `session_summary` | Planner, Trainer |
| Publishing | `publish_model`, `log_to_wandb` | Publisher |
| Training | `run_training_local`, `submit_basilica_job` | (advanced) |
| Utilities | `run_python`, `read_file`, `write_file` | (advanced) |
| Market Data | `get_price`, `get_price_history` | (advanced) |

---

## SN50 Competition Details

### What is Synth (SN50)?

Synth is Bittensor Subnet 50 — a decentralised probabilistic price forecasting network. Miners compete to produce the most well-calibrated price distribution forecasts.

### Prediction Requirements

Each miner must produce:
- **1,000 Monte Carlo price paths** per asset
- **289 timesteps** (24 hours at 5-minute intervals)
- Paths must represent **continuous probability distributions**, not point forecasts

### Tracked Assets

| Asset | Weight |
|-------|--------|
| BTC | 1.00 |
| ETH | 0.67 |
| SOL | 0.59 |
| XAU (gold) | 2.26 |
| SPYX (S&P 500) | 2.99 |
| NVDAX | 1.39 |
| TSLAX | 1.42 |
| AAPLX | 1.86 |
| GOOGLX | 1.43 |

### Scoring: CRPS

Predictions are scored using **CRPS** (Continuous Ranked Probability Score):

```
CRPS = (1/N) * sum(|y_n - x|) - (1/2N^2) * sum(sum(|y_n - y_m|))
```

- Lower CRPS = better calibrated distribution
- Measured at increments: 5, 10, 15, 30, 60, 180, 360, 720, 1440 minutes
- Per-asset scores are weighted by the asset weights above

CRPS rewards distributions that are both **accurate** (centered on the true value) and **sharp** (tight, not overly dispersed).

### Model Components (from open-synth-miner)

**15 Backbone Blocks:**
TransformerBlock, LSTMBlock, GRUBlock, ResConvBlock, FourierBlock, TimesNetBlock, WaveNetBlock, and more — each implementing a different approach to sequence modeling.

**6 Head Types (increasing expressiveness):**
1. **GBMHead** — Geometric Brownian Motion (simplest baseline)
2. **GARCHHead** — GARCH volatility clustering
3. **HestonHead** — Heston stochastic volatility
4. **SDEHead** — Neural SDE (learned drift + diffusion)
5. **FlowHead** — Normalizing flow
6. **NeuralSDEHead** — Full neural SDE (most expressive)

**10 Presets:**
Pre-configured block+head combinations ready to run: `transformer_lstm`, `pure_transformer`, `conv_gru`, `timesnet_fourier`, etc.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'research'"

`open-synth-miner` is not installed. Follow the installation steps in [Installation > Step 4](#4-install-open-synth-miner).

### Agent gets stuck in a loop

The pipeline has built-in stall detection. If the same config is produced twice in a row, a warning is injected. If the problem persists, try:
- Increasing `--retries`
- Starting with a higher `--temperature`
- Running `python main.py sweep` first to establish baseline results the Planner can reference

### CRPS scores are poor

- Increase `RESEARCH_N_PATHS` to 1000 for proper Monte Carlo sampling
- Increase `RESEARCH_EPOCHS` for more training
- Try a more expressive head (SDEHead or NeuralSDEHead instead of GBMHead)
- Increase `RESEARCH_D_MODEL` (64 or 128) for more model capacity

### LLM API errors

- Verify `CHUTES_API_KEY` is set correctly in `.env`
- Check that `CHUTES_BASE_URL` is reachable
- The default models (Qwen/Qwen3-235B-A22B, Qwen/Qwen2.5-Coder-32B-Instruct) must be available on your provider

### Publishing fails

- Ensure `HF_REPO_ID` is set and you have write access to the Hugging Face repo
- Run `huggingface-cli login` to authenticate
- For W&B, run `wandb login` first
