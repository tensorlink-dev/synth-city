# synth-city

Agentic layer for building and training competitive models on **Bittensor Subnet 50 (Synth)** — a decentralised probabilistic price forecasting network.

## Architecture

```
main.py                          CLI entry point
├── pipeline/
│   ├── providers/
│   │   ├── simple_agent.py      ~200-line core agent loop (for-loop, not DAGs)
│   │   └── chutes_client.py     Chutes AI LLM client (OpenAI-compatible)
│   ├── agents/
│   │   ├── base.py              BaseAgentWrapper (composition over inheritance)
│   │   ├── planner.py           Planner — diagnostic + execution plan
│   │   ├── code_checker.py      CodeChecker — validates model output
│   │   ├── debugger.py          Debugger — fixes failed validations
│   │   └── trainer.py           Trainer — fits model parameters
│   ├── tools/
│   │   ├── registry.py          Tool registry with dynamic injection
│   │   ├── file_tools.py        write_file, read_file (code via tools only)
│   │   ├── check_shapes.py      SN50 shape validation
│   │   ├── market_data.py       Price data fetching
│   │   └── training_tools.py    Local + Basilica training execution
│   ├── prompts/
│   │   ├── fragments.py         Composable prompt fragment system
│   │   ├── planner_prompts.py   Phased reasoning prompts
│   │   ├── checker_prompts.py   Validation checklist prompts
│   │   ├── debugger_prompts.py  Error pattern catalog prompts
│   │   └── trainer_prompts.py   Training procedure prompts
│   └── orchestrator.py          Retry loops, temperature escalation, stall detection
├── models/
│   ├── base.py                  BaseForecaster interface
│   ├── gbm.py                   Geometric Brownian Motion (baseline)
│   ├── garch.py                 GARCH / EGARCH / GJR-GARCH
│   ├── stochastic_vol.py        Heston stochastic volatility
│   └── neural/
│       ├── lstm_garch.py        LSTM-GARCH hybrid
│       └── nsvm.py              Neural Stochastic Volatility Model
├── subnet/
│   ├── config.py                SN50 constants
│   ├── miner.py                 Prediction generation + submission
│   └── validator.py             CRPS scoring for local evaluation
├── data/
│   ├── market.py                Market data fetching + caching
│   └── preprocessing.py         Feature engineering utilities
├── compute/
│   └── basilica.py              Basilica decentralised GPU training client
└── config.py                    Environment-based configuration
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
| Stall detection | File hash comparison between debug attempts + CRITICAL WARNING injection |
| Temperature escalation | 0.1 → 0.2 → 0.3... on retry failures |
| Ephemeral compression | Large file reads truncated after N chars |
| Phased prompts | Planner: PHASE 1 DIAGNOSTIC → PHASE 2 EXECUTION |
| Mandatory tool use | CodeChecker/Debugger must call check_shapes before finishing |

## Usage

```bash
# Set up environment
cp .env.example .env
# Edit .env with your Chutes AI and Basilica API keys

# Run the full improvement pipeline
python main.py pipeline --assets BTC,ETH,SOL

# Generate predictions with current best models
python main.py predict --assets BTC,ETH,SOL --horizon 24h

# Evaluate a GARCH model locally
python main.py evaluate --asset BTC --variant GJR-GARCH

# Run a single agent for debugging
python main.py agent --name planner --assets BTC
```

## SN50 Competition

**Synth** (SN50) requires miners to produce **1,000 Monte Carlo price paths** per asset at 5-minute intervals over a 24-hour horizon. Predictions are scored using **CRPS** (Continuous Ranked Probability Score) — a metric that rewards well-calibrated probability distributions, not just accurate point forecasts.

### Supported Assets
BTC, ETH, SOL, XAU (gold), SPYX (S&P 500), NVDAX, TSLAX, AAPLX, GOOGLX

### Infrastructure
- **LLM Inference**: [Chutes AI](https://chutes.ai/) (SN64) — OpenAI-compatible API
- **GPU Training**: [Basilica](https://github.com/tplr-ai/basilica) (SN39) — decentralised compute
