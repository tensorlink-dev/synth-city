```
╔══════════════════════════════════════════════════════════════════════════════════╗
│                                                                                │
│  ███████╗██╗   ██╗███╗   ██╗████████╗██╗  ██╗     ██████╗██╗████████╗██╗   ██╗│
│  ██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██║  ██║    ██╔════╝██║╚══██╔══╝╚██╗ ██╔╝│
│  ███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ███████║    ██║     ██║   ██║    ╚████╔╝ │
│  ╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██║    ██║     ██║   ██║     ╚██╔╝  │
│  ███████║   ██║   ██║ ╚████║   ██║   ██║  ██║    ╚██████╗██║   ██║      ██║   │
│  ╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝     ╚═════╝╚═╝   ╚═╝      ╚═╝   │
│                                                                                │
│                    agentic mining for the bittensor network                     │
│                                                                                │
╚══════════════════════════════════════════════════════════════════════════════════╝
```

**synth-city** is an autonomous AI mining system for [Bittensor](https://bittensor.com/). It deploys a chain of specialized agents that continuously discover, train, debug, and publish probabilistic price forecasting models — competing on **Subnet 50 (Synth)** while leveraging multiple Bittensor subnets for compute, inference, and storage.

No manual model tuning. No babysitting training runs. Point it at the network and let the agents mine.

---

## How It Works

synth-city treats mining as a research loop. A pipeline of AI agents collaborates to find the best model architecture, train it, validate it against subnet specs, fix any issues, and publish it — all autonomously.

```
┌─────────┐    ┌─────────┐    ┌─────────────┐    ┌──────────┐    ┌───────────┐
│ Planner │───▸│ Trainer │───▸│ CodeChecker │───▸│ Debugger │───▸│ Publisher │
└─────────┘    └─────────┘    └─────────────┘    └──────────┘    └───────────┘
 surveys        runs           validates          fixes            publishes
 components     experiments    against SN50       failures         to HF Hub
 + history      on GPUs        specs              + retries        + logs to W&B
```

**Planner** reviews available backbone blocks, prediction heads, and past experiment history from Hippius storage. It designs the next experiment to try.

**Trainer** executes the plan — creating and running experiments using [open-synth-miner](https://github.com/tensorlink-dev/open-synth-miner), a composable PyTorch framework with 15+ backbone blocks and 6 prediction heads.

**CodeChecker** validates the experiment configuration and results against SN50's strict output requirements (1,000 Monte Carlo paths, correct tensor shapes, valid CRPS scores).

**Debugger** catches failures. If training errors out or validation fails, the debugger analyzes the error, patches the config, and retries — with temperature escalation and stall detection to avoid loops.

**Publisher** takes the winning model and pushes it to Hugging Face Hub, logs metrics to Weights & Biases, and persists everything to Hippius for the next pipeline run.

The orchestrator chains these agents with retry logic, so the full cycle runs hands-off. Each agent reasons via LLM inference through **Chutes AI (SN64)**, trains on decentralized GPUs via **Basilica (SN39)**, and stores results on **Hippius** — the entire mining operation runs on Bittensor infrastructure.

---

## Bittensor Subnets

synth-city is built on top of multiple Bittensor subnets, using each for what it does best:

| Subnet | Name | Role | What synth-city uses it for |
|--------|------|------|-----------------------------|
| **SN50** | **Synth** | Target competition | The price forecasting subnet. Miners submit 1,000 Monte Carlo price paths per asset at 5-minute intervals over 24 hours. Scored by CRPS (Continuous Ranked Probability Score). This is what we're mining. |
| **SN39** | **Basilica** | GPU compute | Decentralized GPU marketplace. synth-city rents GPUs (V100, A4000, A6000) to train models remotely, with budget caps and automatic pod management. |
| **SN64** | **Chutes AI** | LLM inference | OpenAI-compatible inference API. Powers all agent reasoning — every planning decision, code review, and debug analysis runs through Chutes. |
| | **Hippius** | Decentralized storage | S3-compatible storage on Bittensor. Persists experiment results, model checkpoints, and pipeline history across runs. The planner loads past results from Hippius to inform future experiments. |

### SN50 — Synth (the target)

synth-city mines Subnet 50 by generating probabilistic price forecasts for 9 assets:

| Asset | Weight | Asset | Weight |
|-------|--------|-------|--------|
| BTC | 1.00 | SPYX (S&P 500) | 2.99 |
| ETH | 0.67 | NVDAX (NVIDIA) | 1.39 |
| SOL | 0.59 | TSLAX (Tesla) | 1.42 |
| XAU (Gold) | 2.26 | AAPLX (Apple) | 1.86 |
| | | GOOGLX (Google) | 1.43 |

Each submission: **1,000 Monte Carlo paths** x **288 timesteps** (5-min intervals over 24h) per asset. Validators score submissions using CRPS — a metric that rewards well-calibrated probability distributions. Lower is better.

---

## Quick Start

```bash
# Clone
git clone https://github.com/tensorlink-dev/synth-city.git
cd synth-city

# Install
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your API keys:
#   CHUTES_API_KEY     — LLM inference (SN64)
#   BASILICA_API_TOKEN — GPU compute (SN39)
#   HIPPIUS_ACCESS_KEY — decentralized storage
#   HIPPIUS_SECRET_KEY
#   BT_WALLET_NAME     — Bittensor wallet
#   BT_HOTKEY_NAME

# Run the full agentic pipeline
synth-city pipeline

# Or use python directly
python main.py pipeline
```

---

## CLI

synth-city ships as a terminal application with Rich-powered output and a gradient banner:

```bash
# Full autonomous pipeline — agents discover, train, validate, debug, publish
synth-city pipeline

# Run a preset sweep — quick benchmarking across known-good architectures
synth-city sweep
synth-city sweep --presets transformer,lstm,wavenet

# Run a single experiment manually
synth-city experiment --blocks TransformerBlock,LSTMBlock --head SDEHead --epochs 5

# One-liner convenience experiment
synth-city quick --blocks TransformerBlock --head GBMHead

# Query experiment history from decentralized storage
synth-city history hippius
synth-city history hippius --run-id latest
synth-city history trackio --trends
synth-city history hf

# Pre-download training data (avoids mid-pipeline downloads)
synth-city data download
synth-city data download --assets BTC,ETH,SOL --timeframe all
synth-city data info

# Run a single agent for debugging
synth-city agent --name planner
synth-city agent --name trainer --message "Try a WaveNet + SDE architecture"

# HTTP bridge server (for OpenClaw integration or external tools)
synth-city bridge
synth-city client blocks
synth-city client run --publish
```

---

## Architecture

### Agent System

Each agent is a thin wrapper around a core loop. Agents subclass `BaseAgentWrapper` and define their own system prompt, toolset, and optional context. The loop is a ~200-line for-loop in `SimpleAgent`: send messages, execute tool calls, append results, repeat until the `finish` tool is called.

```
pipeline/
  orchestrator.py           Retry loops, temperature escalation, stall detection
  providers/
    simple_agent.py         Core agent loop (~200 lines, no DAGs)
    chutes_client.py        OpenAI-compatible LLM client → Chutes AI (SN64)
  agents/
    base.py                 BaseAgentWrapper — composition pattern
    planner.py              Discovers components, loads Hippius history, plans experiments
    trainer.py              Executes experiments via ResearchSession
    code_checker.py         Validates configs + results against SN50 specs
    debugger.py             Fixes failed experiments with error pattern catalog
    publisher.py            HF Hub + W&B + Hippius publishing
    author.py               Writes new backbone blocks/heads into the registry
```

### Tool Registry

Tools auto-register globally via the `@tool` decorator. Each agent gets a scoped subset through `build_toolset()`. Adding a new tool = one decorated function, no config changes needed.

```
pipeline/tools/
  registry.py               @tool decorator, global registry, build_toolset()
  research_tools.py         ResearchSession API (create/run/validate/compare)
  training_tools.py         Local + Basilica (SN39) GPU training
  hippius_store.py          Hippius decentralized storage (S3-compatible)
  publish_tools.py          HF Hub + W&B logging
  market_data.py            Pyth Network price feeds
  check_shapes.py           SN50 shape validation
  file_tools.py             File I/O for agent workspace
  register_tools.py         Write new components + reload registry
  analysis_tools.py         W&B + HF Hub analysis
```

### Prompt Engineering

The sophistication lives in the prompts, not the framework. Each agent has phased reasoning, error catalogs, validation checklists, and component reference docs built into their system prompts.

```
pipeline/prompts/
  fragments.py              Composable prompt building blocks
  planner_prompts.py        Phased reasoning + component reference
  trainer_prompts.py        Experiment execution procedures
  checker_prompts.py        SN50 validation checklist
  debugger_prompts.py       Error pattern catalog
  publisher_prompts.py      Publishing procedure
  author_prompts.py         Component authoring guidelines
```

---

## Bittensor Mining Setup

### Register on SN50

```bash
# Install Bittensor CLI
pip install bittensor

# Create a wallet (if you don't have one)
btcli wallet create --wallet.name default --wallet.hotkey default

# Register on Subnet 50
btcli subnet register --netuid 50 --wallet.name default --wallet.hotkey default
```

### Configure for Mining

All configuration is environment-based via `.env`:

```bash
# Bittensor wallet
BT_WALLET_NAME=default
BT_HOTKEY_NAME=default
BT_NETWORK=finney
BT_NETUID=50

# Chutes AI — LLM inference (SN64)
CHUTES_API_KEY=your-key-here

# Basilica — GPU compute (SN39)
BASILICA_API_TOKEN=your-token-here
BASILICA_MAX_HOURLY_RATE=0.44
BASILICA_ALLOWED_GPU_TYPES=TESLA V100,RTX-A4000,RTX-A6000

# Hippius — decentralized storage
HIPPIUS_ENDPOINT=https://s3.hippius.network
HIPPIUS_ACCESS_KEY=your-access-key
HIPPIUS_SECRET_KEY=your-secret-key
HIPPIUS_BUCKET=synth-city

# Publishing
HF_TOKEN=your-hf-token
HF_REPO_ID=your-username/your-model-repo
WANDB_PROJECT=synth-city
```

### Run the Miner

```bash
# Start the autonomous pipeline — it will loop through agents continuously
synth-city pipeline --publish

# Or with custom settings
synth-city pipeline --retries 10 --temperature 0.1 --publish
```

The pipeline will:
1. **Plan** — survey available blocks/heads, load history from Hippius, design an experiment
2. **Train** — run the experiment on Basilica GPUs (or locally)
3. **Validate** — check outputs against SN50 specs (1,000 paths, 288 timesteps, valid CRPS)
4. **Debug** — if anything fails, diagnose and retry with escalating temperature
5. **Publish** — push the best model to HF Hub and log to W&B

---

## open-synth-miner

synth-city wraps [open-synth-miner](https://github.com/tensorlink-dev/open-synth-miner), a composable PyTorch framework for building probabilistic forecasting models. All backbone blocks share a uniform tensor interface:

```
(batch, seq, d_model) → (batch, seq, d_model)
```

This means blocks can be freely stacked and swapped. The agent pipeline exploits this — the planner can mix and match any combination of blocks and heads to find what works best.

**Backbone blocks**: TransformerBlock, LSTMBlock, WaveNetBlock, TCNBlock, NBeatsBlock, TFTBlock, InformerBlock, AutoformerBlock, FEDformerBlock, PatchTSTBlock, TimesNetBlock, CrossformerBlock, iTransformerBlock, FreTSBlock, MambaBlock

**Prediction heads**: GBMHead, SDEHead, FlowHead, MixtureHead, CopulaHead, QuantileHead

---

## Development

```bash
# Install with dev dependencies
pip install -e .[dev]

# Lint
ruff check .
ruff check --fix .

# Type check
mypy .

# Test
pytest
pytest -k test_name
```

---

## Project Structure

```
synth-city/
├── main.py                     CLI entry point
├── config.py                   Environment-based configuration
├── cli/
│   ├── app.py                  CLI application (synth-city command)
│   └── display.py              Rich-powered terminal output
├── pipeline/
│   ├── orchestrator.py         Agent chaining + retry logic
│   ├── providers/
│   │   ├── simple_agent.py     Core agent loop
│   │   └── chutes_client.py    Chutes AI LLM client (SN64)
│   ├── agents/                 Agent wrappers (planner, trainer, etc.)
│   ├── tools/                  Tool registry + implementations
│   └── prompts/                System prompts for each agent
├── compute/
│   ├── basilica.py             Basilica GPU client (SN39)
│   └── training_server.py      HTTP training server for GPU pods
├── subnet/
│   ├── config.py               SN50-specific configuration
│   ├── miner.py                Bittensor miner implementation
│   └── validator.py            CRPS scoring validator
└── integrations/
    └── openclaw/               HTTP bridge server + CLI client
```

---

## License

MIT
