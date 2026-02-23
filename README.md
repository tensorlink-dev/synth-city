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

**synth-city** is the MLOps, R&D, and model optimization engine for agentic mining on [Bittensor](https://bittensor.com/). It chains specialized AI agents together to continuously discover, train, debug, and publish probabilistic price forecasting models — competing on **Subnet 50 (Synth)** while leveraging multiple Bittensor subnets for GPU compute, LLM inference, and decentralized storage.

synth-city doesn't run in isolation. It's the research backbone that **OpenClaw bots** plug into — giving conversational AI agents the ability to steer research, design experiments, author new model components, reshape the pipeline itself, and deploy winning models to mine on the network. Multiple bots interact with the system concurrently, each with their own isolated session.

---

## The Stack

Three layers work together to turn agentic reasoning into Bittensor mining rewards:

```
┌──────────────────────────────────────────────────────────────────────┐
│  OpenClaw Bots                                                       │
│  Multiple conversational AI agents that steer research, write new    │
│  model components, create new pipeline agents, and deploy models.    │
│  Each bot gets an isolated session. Results ranked across all bots.  │
└──────────────────────┬───────────────────────────────────────────────┘
                       │ HTTP bridge (:8377)
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  synth-city (this repo)                                              │
│  MLOps / R&D engine. Chains specialized agents together:             │
│  Planner → Trainer → CodeChecker → Debugger → Publisher              │
│  Handles orchestration, retries, experiment history, and publishing.  │
└──────────────────────┬───────────────────────────────────────────────┘
                       │ Python API
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  open-synth-miner                                                    │
│  Composable PyTorch framework. 15+ backbone blocks, 6 prediction     │
│  heads, uniform tensor interface. The actual model training happens   │
│  here.                                                               │
└──────────────────────────────────────────────────────────────────────┘
```

### Why this approach?

**[open-synth-miner](https://github.com/tensorlink-dev/open-synth-miner)** provides the composable model building blocks — any backbone block can be stacked with any other because they share a uniform tensor interface `(batch, seq, d_model) → (batch, seq, d_model)`. But choosing *which* blocks to combine, tuning hyperparameters, validating against subnet specs, debugging failures, and publishing models is a research grind.

**synth-city** automates that grind. Its agent pipeline plans experiments based on past results, trains them on decentralized GPUs, validates outputs against SN50 requirements, diagnoses and fixes failures, and publishes the winners — all autonomously.

**OpenClaw bots** add the steering layer. Instead of just letting the pipeline run blind, multiple bots can observe what's happening, redirect research ("try more attention-based architectures"), create manual experiments, inspect results, and decide when to publish to production. Critically, bots can also **write entirely new backbone blocks and prediction heads** into the open-synth-miner registry, **create new pipeline agents** with custom reasoning and toolsets, and **recompose the pipeline itself** — adding, removing, or reordering stages. The system evolves through use.

This separation means:
- The **model framework** stays focused on PyTorch composability
- The **R&D engine** stays focused on orchestration and validation
- The **bots** provide high-level intelligence, human-in-the-loop control, and can extend both the model framework and the pipeline itself
- Each layer evolves independently — bots can ship new model architectures and agents without touching core infrastructure

---

## Bittensor Subnets

The entire mining operation runs on Bittensor infrastructure. synth-city uses four subnets:

| Subnet | Name | Role | What synth-city uses it for |
|--------|------|------|-----------------------------|
| **SN50** | **Synth** | Target competition | The price forecasting subnet we're mining. Miners submit Monte Carlo price paths scored by CRPS. |
| **SN39** | **Basilica** | GPU compute | Decentralized GPU marketplace. Rents V100, A4000, A6000 GPUs for remote model training with budget caps and automatic pod management. |
| **SN64** | **Chutes AI** | LLM inference | OpenAI-compatible inference API. Powers all agent reasoning — every planning decision, code review, and debug analysis runs through Chutes. |
| **SN30** | **Hippius** | Decentralized storage | S3-compatible decentralized storage. Persists experiment results, model checkpoints, and pipeline history across runs. The planner loads past results from Hippius to inform future experiments. |

### SN50 — Synth (the target)

synth-city mines Subnet 50 by generating probabilistic price forecasts across two timeframes:

**Standard (24h)** — 288 steps at 5-minute intervals
**HFT (1h)** — 60 steps at 1-minute intervals

Each submission requires **1,000 Monte Carlo price paths** per asset per timeframe. Validators score submissions using **CRPS** (Continuous Ranked Probability Score) — a metric that rewards well-calibrated probability distributions. Lower is better.

**Assets and scoring weights:**

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
| GOOGLX | 1.43 | Alphabet/Google |

Higher-weighted assets (SPYX, XAU, AAPLX) have more impact on overall miner ranking.

---

## How the R&D Pipeline Works

synth-city treats mining as a continuous research loop. A chain of AI agents collaborates to find the best model, validate it, fix it if needed, and ship it:

```
┌─────────┐    ┌─────────┐    ┌─────────────┐    ┌──────────┐    ┌───────────┐
│ Planner │───▸│ Trainer │───▸│ CodeChecker │───▸│ Debugger │───▸│ Publisher │
└─────────┘    └─────────┘    └─────────────┘    └──────────┘    └───────────┘
 surveys        runs           validates          fixes            publishes
 components     experiments    against SN50       failures         to HF Hub
 + Hippius      on Basilica    specs              + retries        + Hippius
 history        GPUs (SN39)                                        (SN30)
```

**Planner** — Reviews available backbone blocks, prediction heads, and past experiment history from Hippius (SN30). Designs the next experiment based on what's worked and what hasn't.

**Trainer** — Executes the plan using open-synth-miner's `ResearchSession` API. Trains on decentralized GPUs via Basilica (SN39) or locally.

**CodeChecker** — Validates experiment configuration and results against SN50's strict output requirements (1,000 paths, correct tensor shapes, valid CRPS).

**Debugger** — If training fails or validation doesn't pass, the debugger analyzes the error, patches the config, and retries with temperature escalation and stall detection to avoid loops.

**Publisher** — Pushes the winning model to Hugging Face Hub, logs metrics to Weights & Biases, and persists everything to Hippius (SN30) for the next pipeline run.

All agent reasoning runs through Chutes AI (SN64). The orchestrator chains agents with retry logic and temperature escalation so the full cycle runs hands-off.

---

## OpenClaw Bots

synth-city is the R&D engine. OpenClaw bots are the operators. They connect via an HTTP bridge server, and multiple bots can interact with the system at the same time — each with an isolated session, workspace, and pipeline state.

### Steer research

Bots direct the research process through conversation:
- Run the full autonomous pipeline or kick off individual experiments
- Choose which backbone blocks and prediction heads to try
- Review past results from Hippius (SN30) and guide the planner toward promising architectures
- Compare experiment rankings across all active bot sessions
- Query live market prices and historical data

### Write new model components

Bots don't just use the existing blocks and heads — they can author new ones:
- Write new **backbone blocks** into the open-synth-miner component registry
- Write new **prediction heads** with custom architectures
- Study existing component source code to learn the tensor interface
- Reload the registry so new components are immediately available for experiments

### Create new agents and reshape the pipeline

Bots can extend synth-city itself:
- Write entirely new **pipeline agents** (subclass `BaseAgentWrapper`, define prompts and tools)
- Write new **prompt modules** with custom reasoning strategies
- **Recompose the pipeline** — add, remove, or reorder stages via the PipelineArchitect agent
- Tune orchestrator recovery parameters (retries, temperature escalation, stall detection)

### Deploy models for mining

When a bot is happy with results, it publishes:
- Push winning models to **Hugging Face Hub** for production mining
- Log metrics to **Weights & Biases** for tracking
- Persist full experiment history to **Hippius (SN30)** for future pipeline runs

### Multi-bot concurrency

Multiple OpenClaw bots work the system in parallel:
- Each bot gets an **isolated session** — its own workspace, experiments, and pipeline state
- Up to **10 concurrent pipeline runs** across all bots (configurable)
- **Cross-bot comparison** — rank experiments from all active sessions together
- Idle sessions are automatically reaped after TTL expiry
- Per-bot file I/O prevents workspace conflicts

### Connecting bots

```bash
# Start the bridge on your GPU server
synth-city bridge

# Option A: SSH tunnel (recommended — encrypted, no config changes)
ssh -L 8377:localhost:8377 user@gpu-server -N

# Option B: Direct network + API key
# GPU server .env:  BRIDGE_HOST=0.0.0.0  BRIDGE_API_KEY=your-secret-key
# Bot machine:      export SYNTH_BRIDGE_URL=http://<gpu-server-ip>:8377
#                   export BRIDGE_API_KEY=your-secret-key
```

The synth-city skill is available on ClawHub for OpenClaw agents:

```bash
python integrations/openclaw/setup.py      # install into OpenClaw workspace
python integrations/openclaw/publish.py     # publish to ClawHub
```

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
#   HIPPIUS_ACCESS_KEY — decentralized storage (SN30)
#   HIPPIUS_SECRET_KEY
#   BT_WALLET_NAME     — Bittensor wallet
#   BT_HOTKEY_NAME

# Run the full agentic pipeline
synth-city pipeline

# Or start the bridge for OpenClaw bots
synth-city bridge
```

---

## CLI

```bash
# Full autonomous pipeline — agents discover, train, validate, debug, publish
synth-city pipeline
synth-city pipeline --publish --retries 10

# Preset sweep — benchmark across known-good architectures
synth-city sweep
synth-city sweep --presets transformer,lstm,wavenet

# Single experiment
synth-city experiment --blocks TransformerBlock,LSTMBlock --head SDEHead --epochs 5

# One-liner convenience experiment
synth-city quick --blocks TransformerBlock --head GBMHead

# Experiment history from Hippius (SN30)
synth-city history hippius
synth-city history hippius --run-id latest
synth-city history trackio --trends
synth-city history hf

# Pre-download training data
synth-city data download
synth-city data download --assets BTC,ETH,SOL --timeframe all
synth-city data info

# Run a single agent
synth-city agent --name planner
synth-city agent --name trainer --message "Try a WaveNet + SDE architecture"

# HTTP bridge for OpenClaw bots
synth-city bridge
synth-city client blocks
synth-city client run --publish
```

---

## Mining Setup

### Register on SN50

```bash
pip install bittensor
btcli wallet create --wallet.name default --wallet.hotkey default
btcli subnet register --netuid 50 --wallet.name default --wallet.hotkey default
```

### Configure

All settings via `.env`:

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

# Hippius — decentralized storage (SN30)
HIPPIUS_ENDPOINT=https://s3.hippius.network
HIPPIUS_ACCESS_KEY=your-access-key
HIPPIUS_SECRET_KEY=your-secret-key
HIPPIUS_BUCKET=synth-city

# Publishing
HF_TOKEN=your-hf-token
HF_REPO_ID=your-username/your-model-repo
WANDB_PROJECT=synth-city
```

### Run

```bash
# Autonomous pipeline
synth-city pipeline --publish

# Or let OpenClaw bots drive
synth-city bridge
```

---

## open-synth-miner

synth-city wraps [open-synth-miner](https://github.com/tensorlink-dev/open-synth-miner), a composable PyTorch framework for building probabilistic forecasting models. All backbone blocks share a uniform tensor interface:

```
(batch, seq, d_model) → (batch, seq, d_model)
```

Blocks can be freely stacked and swapped. The agent pipeline exploits this — the planner mixes and matches any combination of blocks and heads to find what works best. OpenClaw bots can write new blocks and heads directly into the registry, extending the search space without touching core framework code.

**Backbone blocks** (15): TransformerBlock, LSTMBlock, WaveNetBlock, TCNBlock, NBeatsBlock, TFTBlock, InformerBlock, AutoformerBlock, FEDformerBlock, PatchTSTBlock, TimesNetBlock, CrossformerBlock, iTransformerBlock, FreTSBlock, MambaBlock

**Prediction heads** (6): GBMHead, SDEHead, FlowHead, MixtureHead, CopulaHead, QuantileHead

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
│   ├── orchestrator.py         Agent chaining, retries, temperature escalation
│   ├── providers/
│   │   ├── simple_agent.py     Core agent loop (~200 lines, no DAGs)
│   │   └── chutes_client.py    Chutes AI LLM client (SN64)
│   ├── agents/                 Planner, Trainer, CodeChecker, Debugger, Publisher, Author
│   ├── tools/                  Tool registry + implementations
│   │   ├── research_tools.py   ResearchSession API
│   │   ├── training_tools.py   Local + Basilica GPU training (SN39)
│   │   ├── hippius_store.py    Decentralized storage (SN30)
│   │   ├── market_data.py      Pyth Network price feeds
│   │   ├── check_shapes.py     SN50 shape validation
│   │   └── ...
│   └── prompts/                System prompts for each agent
├── compute/
│   ├── basilica.py             Basilica GPU client (SN39)
│   └── training_server.py      HTTP training server for GPU pods
├── subnet/
│   ├── config.py               SN50-specific configuration
│   ├── miner.py                Bittensor miner implementation
│   └── validator.py            CRPS scoring validator
└── integrations/
    └── openclaw/
        ├── bridge.py           HTTP bridge server (50+ endpoints)
        ├── bot_sessions.py     Per-bot session isolation + TTL reaper
        ├── client.py           Python client for the bridge
        └── skill/              OpenClaw skill (SKILL.md + tools)
```

---

## Development

```bash
pip install -e .[dev]
ruff check .            # lint
mypy .                  # type check
pytest                  # test
```

---

## License

MIT
