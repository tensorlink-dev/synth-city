# synth-city

Agentic layer for building and training competitive models on **Bittensor Subnet 50 (Synth)** — a decentralised probabilistic price forecasting network. Built on top of [`open-synth-miner`](https://github.com/tensorlink-dev/open-synth-miner).

## Architecture

```
main.py                              CLI entry point
├── pipeline/
│   ├── providers/
│   │   ├── simple_agent.py          ~200-line core agent loop (for-loop, not DAGs)
│   │   └── chutes_client.py         Chutes AI LLM client (OpenAI-compatible)
│   ├── agents/
│   │   ├── base.py                  BaseAgentWrapper (composition over inheritance)
│   │   ├── planner.py               Planner — discovers components, produces experiment plan
│   │   ├── code_checker.py          CodeChecker — validates experiment configs + results
│   │   ├── debugger.py              Debugger — fixes failed experiments
│   │   ├── trainer.py               Trainer — executes experiments via ResearchSession
│   │   └── publisher.py             Publisher — HF Hub + W&B production tracking
│   ├── tools/
│   │   ├── registry.py              Tool registry with dynamic injection
│   │   ├── research_tools.py        ResearchSession API (list/create/validate/run/compare)
│   │   ├── publish_tools.py         HF Hub publishing + W&B logging
│   │   ├── file_tools.py            write_file, read_file (code via tools only)
│   │   ├── check_shapes.py          SN50 shape validation
│   │   ├── market_data.py           Price data fetching
│   │   └── training_tools.py        Local + Basilica training execution
│   ├── prompts/
│   │   ├── fragments.py             Composable prompt fragment system
│   │   ├── planner_prompts.py       Phased reasoning + component reference
│   │   ├── checker_prompts.py       Validation checklist prompts
│   │   ├── debugger_prompts.py      Error pattern catalog prompts
│   │   ├── trainer_prompts.py       Experiment execution prompts
│   │   └── publisher_prompts.py     Publishing procedure prompts
│   └── orchestrator.py              Retry loops, temperature escalation, stall detection
├── models/                          Standalone model implementations (fallback)
│   ├── base.py                      BaseForecaster interface
│   ├── gbm.py                       Geometric Brownian Motion (baseline)
│   ├── garch.py                     GARCH / EGARCH / GJR-GARCH
│   └── stochastic_vol.py            Heston stochastic volatility
├── subnet/
│   ├── config.py                    SN50 constants
│   ├── miner.py                     Prediction generation + submission
│   └── validator.py                 CRPS scoring for local evaluation
├── data/
│   ├── market.py                    Market data fetching + caching
│   └── preprocessing.py             Feature engineering utilities
├── compute/
│   └── basilica.py                  Basilica decentralised GPU training client
├── integrations/openclaw/
│   ├── bridge.py                    HTTP bridge server (standalone or OpenClaw backend)
│   ├── client.py                    CLI client for the bridge
│   ├── setup.py                     Setup utilities
│   └── skill/                       OpenClaw skill definitions
└── config.py                        Environment-based configuration
```

## How It Works

The agentic layer wraps `open-synth-miner`'s `ResearchSession` API, which provides:
- **15 composable backbone blocks** (Transformer, LSTM, GRU, ResConv, Fourier, TimesNet, etc.)
- **6 head types** (GBMHead → NeuralSDEHead, increasing expressiveness)
- **10 ready-to-run presets** (transformer_lstm, pure_transformer, conv_gru, etc.)
- **Zero-side-effect experiment execution** with CRPS scoring

The agents use these components through tool calls:

```
Planner:   list_blocks → list_heads → list_presets → session_summary → produce plan
Trainer:   create_experiment → run_experiment → compare_results → report best
Checker:   validate_experiment → describe_experiment → pass/fail
Debugger:  create_experiment (fixed) → validate_experiment → re-run
Publisher: validate_experiment → publish_model → log_to_wandb
```

## Core Philosophy

**Bitter lesson of agent frameworks**: keep the framework minimal, invest in prompt engineering and context management.

- **SimpleAgent** is a ~200-line for-loop: send messages → execute tool calls → append results → repeat until `finish`.
- Sophistication lives in the **prompts** (phased reasoning, error catalogs, checklists) and **orchestration** (retry loops, temperature escalation, stall detection).
- Composition over inheritance: agents are thin wrappers adding context, not complex class hierarchies.

## Key Patterns

| Pattern | Implementation |
|---------|---------------|
| Per-agent model selection | `<AGENT>_MODEL` env vars → different LLMs per agent |
| Tool injection | Tools dynamically scoped per agent via registry |
| Argument coercion | Handles sloppy LLM output (empty string → empty list, JSON-in-string, etc.) |
| Stall detection | Experiment config comparison between debug attempts + CRITICAL WARNING injection |
| Temperature escalation | 0.1 → 0.2 → 0.3... on retry failures |
| Ephemeral compression | Large tool outputs truncated after N chars |
| Phased prompts | Planner: PHASE 1 DIAGNOSTIC → PHASE 2 EXECUTION |
| Mandatory tool use | CodeChecker must call validate_experiment before finishing |

## Usage

```bash
# Install open-synth-miner
pip install -e /path/to/open-synth-miner

# Set up environment
cp .env.example .env
# Edit .env with your Chutes AI API key

# Run the full agentic pipeline (Planner → Trainer → Checker → Debugger)
python main.py pipeline

# Run with auto-publish to HF Hub
python main.py pipeline --publish

# Quick baseline sweep (no agents, direct ResearchSession)
python main.py sweep
python main.py sweep --presets transformer_lstm,pure_transformer,conv_gru

# Run a single experiment
python main.py experiment --blocks TransformerBlock,LSTMBlock --head SDEHead --d-model 64

# One-liner convenience
python main.py quick --blocks TransformerBlock,LSTMBlock

# Start the HTTP bridge server
python main.py bridge
python main.py bridge --port 9000

# Talk to the bridge from the CLI (no OpenClaw needed)
python main.py client blocks
python main.py client heads
python main.py client presets
python main.py client price BTC
python main.py client history ETH 30
python main.py client run --publish

# Run a single agent for debugging
python main.py agent --name planner
python main.py agent --name trainer
```

## HTTP Bridge

The bridge server exposes synth-city's pipeline and research tools as a lightweight HTTP API. It works standalone (curl, any HTTP client) or as a backend for agent frameworks like [OpenClaw](https://github.com/openclaw).

```bash
python main.py bridge                # default: 127.0.0.1:8377
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/pipeline/run` | Kick off a full pipeline run |
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

The built-in CLI client (`python main.py client <action>`) wraps these endpoints for quick terminal use.

## SN50 Competition

**Synth** (SN50) requires miners to produce **1,000 Monte Carlo price paths** per asset at 5-minute intervals over a 24-hour horizon. Predictions are scored using **CRPS** (Continuous Ranked Probability Score) — a metric that rewards well-calibrated probability distributions, not just accurate point forecasts.

### Supported Assets
BTC, ETH, SOL, XAU (gold), SPYX (S&P 500), NVDAX, TSLAX, AAPLX, GOOGLX

### Infrastructure
- **Model Research**: [open-synth-miner](https://github.com/tensorlink-dev/open-synth-miner) — composable PyTorch research framework
- **LLM Inference**: [Chutes AI](https://chutes.ai/) (SN64) — OpenAI-compatible API
- **GPU Training**: [Basilica](https://github.com/tplr-ai/basilica) (SN39) — decentralised compute
- **Model Publishing**: Hugging Face Hub + Weights & Biases
