<div align="center">

```
0101   _________________________________________________________________________________________   1011
110    |                                                                                       |    010
01     |          _                _          _          _          _                _         |     11
10     |         | |        _     | |        / \        | |     _  | |        _     | |        |     01
00     |      _  | |  _    | |  _ | | _     |   |     _ | | _  | | | |  _    | |  _ | | _      |     10
11     |     | |_| |_| |__| |_| || || |_   _|   |_   _| || || |_| |_| |__| |_| |__| || |       |     11
01     |     |  _   _   _   _  | || ||  _| |     | |_  | || ||  _   _   _   _  | || || |       |     00
110    |     | |_| |_| |_| |_| |_||_|| |___|     |___| |_||_|| |_| |_| |_| |_| |_||_|| |       |    110
0101   |_______________________________________________________________________________________|   1011

                _____ __     __ _   _  _______  _    _    _____  _____  _______ __     __
               / ____|\ \   / /| \ | ||__   __|| |  | |  / ____||_   _||__   __|\ \   / /
              | (___   \ \_/ / |  \| |   | |   | |__| | | |       | |     | |    \ \_/ /
               \___ \   \   /  | . ` |   | |   |  __  | | |       | |     | |     \   /
               ____) |   | |   | |\  |   | |   | |  | | | |____  _| |_    | |      | |
              |_____/    |_|   |_| \_|   |_|   |_|  |_|  \_____||_____|   |_|      |_|
               [ S Y N T H   C I T Y  :  A G E N T I C   M I N I N G   F O R   S N 5 0 ]
```

</div>

**synth-city** is an autonomous AI research pipeline that discovers, trains, debugs, and publishes probabilistic price forecasting models — without manual intervention. Built for [Bittensor Subnet 50 (Synth)](https://github.com/tensorlink-dev/open-synth-miner), it chains specialized AI agents together to iterate through 900+ neural architecture combinations, evaluate them with CRPS scoring, and ship the best model to Hugging Face Hub ready to serve live predictions.

Instead of hand-tuning model configs and running experiments one by one, you point synth-city at SN50 and walk away. A Planner agent surveys available components and prior results, a Trainer executes experiments, a Checker validates outputs, a Debugger fixes failures, and a Publisher pushes the winning model to production. The entire loop retries intelligently — escalating temperature, detecting stalls, and compressing context between stages.

Under the hood it wraps [`open-synth-miner`](https://github.com/tensorlink-dev/open-synth-miner), a composable PyTorch framework providing 15 backbone blocks, 6 prediction heads, and 10 battle-tested presets. synth-city turns that research toolkit into a closed-loop autonomous system.

## SN50 Competition

**Synth** (SN50) requires miners to produce **1,000 Monte Carlo price paths** per asset at **1-minute and 5-minute intervals** over a 24-hour horizon. Predictions are scored using **CRPS** (Continuous Ranked Probability Score) — a metric that rewards well-calibrated probability distributions, not just accurate point forecasts.

### Supported Assets
BTC, ETH, SOL, XAU (gold), SPYX (S&P 500), NVDAX, TSLAX, AAPLX, GOOGLX

### Infrastructure
- **Model Research**: [open-synth-miner](https://github.com/tensorlink-dev/open-synth-miner) — composable PyTorch research framework
- **LLM Inference**: [Chutes AI](https://chutes.ai/) (SN64) — OpenAI-compatible API
- **GPU Training**: [Basilica](https://github.com/tplr-ai/basilica) (SN39) — decentralised compute
- **Model Publishing**: Hugging Face Hub + Weights & Biases

## How It Works

Everything is **registry-based**. Both the research components and the agent tools use registries — you can use built-in blocks out of the box or define and register your own.

### Component Registry (open-synth-miner)

[`open-synth-miner`](https://github.com/tensorlink-dev/open-synth-miner) uses **decorator-based registration** — drop a new block into `src/models/components/` and it's auto-discovered at runtime. No code changes needed. You can also define new hybrid recipes declaratively via YAML configs in `configs/model/`.

Out of the box it ships with:

- **15 backbone blocks** — Transformer, LSTM, GRU, ResConv, Fourier, TimesNet, BiTCN, DLinear, RevIN, and more
- **6 head types** — from `GBMHead` (constant drift/vol baseline) to `NeuralSDEHead` (full neural SDE)
- **10 ready-to-run presets** — tested combinations like `transformer_lstm`, `pure_transformer`, `conv_gru`

All blocks share a uniform tensor interface `(batch, seq, d_model) → (batch, seq, d_model)`, so any block can be stacked with any other block and wired to any head. Custom blocks just need to follow the same contract.

### Tool Registry

Agent tools are plain `@tool`-decorated functions that auto-register with a global registry. Each agent gets a scoped subset of tools injected at runtime via `build_toolset()`. Adding a new tool is one decorated function — no config files, no plumbing.

### Agent Pipeline

The agents use these registries through tool calls:

```
Planner:   list_blocks → list_heads → list_presets → session_summary → produce plan
Trainer:   create_experiment → run_experiment → compare_results → report best
Checker:   validate_experiment → describe_experiment → pass/fail
Debugger:  create_experiment (fixed) → validate_experiment → re-run
Publisher: validate_experiment → publish_model → log_to_trackio
Author:    list_component_files → read_component → write_component → reload_registry → verify
Designer:  list_agents → read_agent → list_available_tools → write_agent_prompt → write_agent → verify
```

#### Agent Designer

The **Agent Designer** is a meta-agent that creates new pipeline agents programmatically. Instead of manually writing agent classes and prompt modules, you describe what you need and the Designer studies existing agents, selects appropriate tools from the registry, writes the prompt module (using the composable `register_fragment()` system), and generates the agent class — all following the project's composition patterns. New agents are immediately runnable via `python main.py agent --name <new_agent>`.

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

# Create a custom block via the ComponentAuthor agent
python main.py agent --name author --message "Write a WaveletBlock that uses wavelet decomposition"

# Design a new pipeline agent via the Agent Designer
python main.py agent --name agent_designer --message "Create an evaluator agent that compares models across assets"

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
│   │   ├── publisher.py             Publisher — HF Hub + W&B production tracking
│   │   ├── author.py               ComponentAuthor — writes new blocks/heads into the registry
│   │   └── agent_designer.py      AgentDesigner — creates new pipeline agents programmatically
│   ├── tools/
│   │   ├── registry.py              Tool registry with dynamic injection
│   │   ├── research_tools.py        ResearchSession API (list/create/validate/run/compare)
│   │   ├── publish_tools.py         HF Hub publishing + W&B logging
│   │   ├── file_tools.py            write_file, read_file (code via tools only)
│   │   ├── check_shapes.py          SN50 shape validation
│   │   ├── market_data.py           Price data fetching
│   │   ├── training_tools.py        Local + Basilica training execution
│   │   ├── register_tools.py       Write components + reload registry
│   │   └── agent_tools.py          Agent design tools (list/read/write agents + prompts)
│   ├── prompts/
│   │   ├── fragments.py             Composable prompt fragment system
│   │   ├── planner_prompts.py       Phased reasoning + component reference
│   │   ├── checker_prompts.py       Validation checklist prompts
│   │   ├── debugger_prompts.py      Error pattern catalog prompts
│   │   ├── trainer_prompts.py       Experiment execution prompts
│   │   ├── publisher_prompts.py     Publishing procedure prompts
│   │   ├── author_prompts.py       Component authoring guidelines
│   │   └── agent_designer_prompts.py  Agent designer workflow + contracts
│   └── orchestrator.py              Retry loops, temperature escalation, stall detection
├── subnet/
│   ├── config.py                    SN50 constants
│   ├── miner.py                     Prediction generation + submission
│   └── validator.py                 CRPS scoring for local evaluation
├── compute/
│   └── basilica.py                  Basilica decentralised GPU training client
├── integrations/openclaw/
│   ├── bridge.py                    HTTP bridge server (standalone or OpenClaw backend)
│   ├── client.py                    CLI client for the bridge
│   ├── setup.py                     Setup utilities
│   └── skill/                       OpenClaw skill definitions
└── config.py                        Environment-based configuration
```
