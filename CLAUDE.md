# CLAUDE.md — synth-city

## What is this project?

synth-city is an autonomous AI research pipeline that discovers, trains, debugs, and publishes probabilistic price forecasting models for Bittensor Subnet 50 (Synth). It wraps [open-synth-miner](https://github.com/tensorlink-dev/open-synth-miner) (a composable PyTorch framework) and chains specialized agents together: Planner → Trainer → CodeChecker → Debugger → Publisher.

## Quick reference

```bash
# Install
pip install -e .            # core dependencies
pip install -e .[dev]       # + pytest, ruff, mypy

# Lint
ruff check .                # lint entire project
ruff check --fix .          # auto-fix

# Type check
mypy .

# Test
pytest                      # runs tests/ (configured in pyproject.toml)
pytest -k <name>            # run a specific test

# Run the full pipeline
python main.py pipeline

# Run a single agent
python main.py agent --name planner

# Pre-download training data (avoids downloading mid-pipeline)
synth-city data download                        # all assets, 5m timeframe
synth-city data download --assets BTC           # BTC only
synth-city data download --assets BTC,ETH,SOL --timeframe all
synth-city data info                            # show loader config and status
```

## Project structure

```
main.py                     CLI entry point (8 subcommands)
config.py                   Env-based config (all settings via .env)
pipeline/
  orchestrator.py           Retry loops, temperature escalation, stall detection
  providers/
    simple_agent.py         ~200-line core agent loop (for-loop, not DAGs)
    chutes_client.py        OpenAI-compatible LLM client (Chutes AI)
  agents/
    base.py                 BaseAgentWrapper — composition pattern
    planner.py              Discovers components, produces experiment plan
    trainer.py              Executes experiments via ResearchSession
    code_checker.py         Validates experiment configs + results
    debugger.py             Fixes failed experiments
    publisher.py            HF Hub + W&B production tracking
    author.py               Writes new blocks/heads into registry
  tools/
    registry.py             @tool decorator, global registry, build_toolset()
    research_tools.py       ResearchSession API (create/run/validate/compare)
    publish_tools.py        HF Hub + W&B logging
    file_tools.py           write_file, read_file
    check_shapes.py         SN50 shape validation
    market_data.py          Price data fetching
    training_tools.py       Local + Basilica training
    register_tools.py       Write components + reload registry
    analysis_tools.py       W&B + HF Hub analysis
    hippius_store.py        S3-compatible decentralised storage
  prompts/
    fragments.py            Composable prompt building blocks
    planner_prompts.py      Phased reasoning + component reference
    checker_prompts.py      Validation checklist
    debugger_prompts.py     Error pattern catalog
    trainer_prompts.py      Experiment execution
    publisher_prompts.py    Publishing procedure
    author_prompts.py       Component authoring guidelines
compute/basilica.py         Decentralised GPU client (SN39)
subnet/                     Bittensor SN50 config, miner, validator (CRPS scoring)
integrations/openclaw/      HTTP bridge server + CLI client
```

## Architecture patterns

- **Tool registry**: `@tool` decorator in `pipeline/tools/registry.py`. Tools auto-register globally; each agent gets a scoped subset via `build_toolset()`. Adding a tool = one decorated function, no config changes.
- **Agent composition**: Agents subclass `BaseAgentWrapper` and override `build_system_prompt()`, `build_tools()`, and optionally `build_context()`. They are thin wrappers, not deep hierarchies.
- **SimpleAgent core**: The agent loop is a ~200-line for-loop in `pipeline/providers/simple_agent.py`. Send messages → execute tool calls → append results → repeat until `finish` tool is called.
- **Argument coercion**: `_coerce_args()` in `simple_agent.py` handles sloppy LLM outputs (empty string → empty list, JSON-in-string, string booleans).
- **Orchestration**: `PipelineOrchestrator` chains agents with retry loops, temperature escalation (0.1 → 0.2 → ...), and stall detection (compares experiment configs between debug attempts).
- **Prompt engineering**: Sophistication lives in prompts (phased reasoning, error catalogs, checklists) in `pipeline/prompts/`, not in framework code.

## Code conventions

- Python 3.10+, `from __future__ import annotations` everywhere
- Type hints on all function signatures (dict, list generics use `dict[str, Any]` not `Dict`)
- Ruff: line length 100, rules E/F/W/I
- Composition over inheritance
- Environment-based config via `python-dotenv` — all settings in `.env`, accessed through `config.py`
- Per-agent model selection: `PLANNER_MODEL`, `TRAINER_MODEL`, etc. env vars

## Key domain concepts

- **CRPS** (Continuous Ranked Probability Score): The competition metric. Lower is better. Rewards well-calibrated probability distributions.
- **SN50**: Bittensor Subnet 50 (Synth). Requires 1,000 Monte Carlo price paths per asset at 5-min intervals over 24 hours.
- **Assets**: BTC, ETH, SOL, XAU, SPYX, NVDAX, TSLAX, AAPLX, GOOGLX (with scoring weights in `config.py`)
- **open-synth-miner**: External composable PyTorch framework. 15 backbone blocks, 6 prediction heads, 10 presets. Blocks share a uniform tensor interface `(batch, seq, d_model) → (batch, seq, d_model)`.
- **ResearchSession**: The API wrapper for running experiments via open-synth-miner (in `pipeline/tools/research_tools.py`)

## Adding new components

**New tool**: Create a function with `@tool` decorator in `pipeline/tools/`. Add the tool name to the relevant agent's `build_tools()` method.

**New agent**: Subclass `BaseAgentWrapper` in `pipeline/agents/`. Implement `agent_name`, `build_system_prompt()`, and `build_tools()`. Add a prompt module in `pipeline/prompts/`.

**New backbone block/head**: Use the ComponentAuthor agent (`python main.py agent --name author`) or manually add to `open-synth-miner`'s registry.
