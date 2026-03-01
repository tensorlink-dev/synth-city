# Development Guide

This guide covers setting up a development environment, running tests, code style, CI/CD, extending synth-city, and contributing.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Tested on 3.10, 3.11, 3.12 |
| Git | Any recent | For cloning repos |
| pip | Recent | Package installation |
| Chutes AI API key | — | Required for pipeline runs ([chutes.ai](https://chutes.ai/)) |

Optional:

| Requirement | Needed for |
|-------------|------------|
| CUDA-capable GPU | Faster local training |
| Docker | Building GPU images |
| Basilica API key | Remote GPU training |
| HuggingFace token | Publishing models |

---

## Environment Setup

### 1. Clone and install

```bash
git clone https://github.com/tensorlink-dev/synth-city.git
cd synth-city
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The `[dev]` extra installs pytest, ruff, and mypy alongside the core dependencies.

### 2. Install open-synth-miner

synth-city's research tools depend on open-synth-miner. For development, an editable install is recommended:

```bash
git clone https://github.com/tensorlink-dev/open-synth-miner.git ../open-synth-miner
pip install -e ../open-synth-miner
```

Alternatively, install directly from GitHub:

```bash
pip install "open-synth-miner @ git+https://github.com/tensorlink-dev/open-synth-miner.git"
```

### 3. Verify installation

```bash
python -c "from research import ResearchSession; print('open-synth-miner OK')"
python -c "from pipeline.orchestrator import PipelineOrchestrator; print('synth-city OK')"
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with at minimum: CHUTES_API_KEY
```

See [Configuration](configuration.md) for all available variables.

---

## Code Conventions

synth-city follows a consistent set of conventions across the codebase:

### Python Style

- **Python 3.10+** — all files use `from __future__ import annotations`
- **Type hints** on all function signatures using modern syntax: `dict[str, Any]` (not `Dict`), `list[str]` (not `List`), `str | None` (not `Optional[str]`)
- **Composition over inheritance** — agents are thin wrappers, tools are standalone functions
- **Environment-based config** — all settings via `.env` and `config.py`, no hard-coded values

### Formatting Rules (Ruff)

| Rule | Value |
|------|-------|
| Line length | 100 characters |
| Python target | 3.10 |
| Enabled rule sets | E (pycodestyle errors), F (pyflakes), W (pycodestyle warnings), I (isort) |

The ruff configuration lives in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
```

### Import Order

Ruff's `I` rule enforces import sorting. Imports should follow the standard grouping:

1. Standard library
2. Third-party packages
3. Local modules

---

## Linting

```bash
# Check for lint errors
ruff check .

# Auto-fix what can be fixed
ruff check --fix .

# Check a specific file
ruff check pipeline/orchestrator.py
```

Ruff is fast — it lints the entire project in under a second.

---

## Type Checking

```bash
mypy .
```

mypy is configured in `pyproject.toml` with `ignore_missing_imports = true` because many dependencies (basilica-sdk, wandb, etc.) lack type stubs. The goal is to catch type errors in synth-city's own code, not in third-party libraries.

```toml
[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
```

---

## Testing

### Running Tests

```bash
# Full test suite
pytest

# Specific test
pytest -k test_monitor

# Verbose output
pytest -v

# Short traceback
pytest --tb=short -q
```

### Test Configuration

Tests are configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

The `asyncio_mode = "auto"` setting means async test functions are automatically detected and run with asyncio — no `@pytest.mark.asyncio` decorator needed.

### Test Structure

Tests live in the `tests/` directory. The current test suite covers:

- `test_monitor.py` — pipeline event monitoring
- `test_openclaw.py` — bridge server and client integration

### Writing New Tests

Follow these patterns:

```python
from __future__ import annotations

def test_something_specific():
    """Test description that explains the scenario."""
    # Arrange
    input_data = {"blocks": "TransformerBlock", "head": "GBMHead"}

    # Act
    result = some_function(input_data)

    # Assert
    assert result["status"] == "ok"
    assert "crps" in result["metrics"]
```

For async tests:

```python
async def test_async_operation():
    result = await async_function()
    assert result is not None
```

---

## CI/CD

### GitHub Actions Workflows

#### `ci.yml` — Main CI Pipeline

Triggers on push to `main` and pull requests to `main`.

| Job | Runner | Python | What it does |
|-----|--------|--------|-------------|
| `lint` | ubuntu-latest | 3.12 | `ruff check .` |
| `typecheck` | ubuntu-latest | 3.12 | `pip install -e .[dev]` then `mypy .` |
| `test` | ubuntu-latest | 3.10, 3.11, 3.12 | `pip install -e .[dev]` then `pytest --tb=short -q` |

All three jobs must pass before a PR can be merged.

#### `docker-gpu.yml` — GPU Image Build

Triggers on push to `main` when `Dockerfile.gpu` or `compute/training_server.py` changes. Can also be triggered manually via `workflow_dispatch`.

Builds and pushes the GPU training image to `ghcr.io/{repo_owner}/synth-city-gpu` with `latest` and SHA tags.

### Running CI Locally

Reproduce CI checks before pushing:

```bash
ruff check .
mypy .
pytest --tb=short -q
```

---

## Extending synth-city

### Adding a New Tool

Tools are the primary extension point. A new tool is a single decorated function:

1. Create a function with the `@tool` decorator:

```python
# pipeline/tools/my_tools.py
from __future__ import annotations
from pipeline.tools.registry import tool

@tool(description="Compute Sharpe ratio from experiment metrics")
def compute_sharpe_ratio(returns_json: str, risk_free_rate: float = 0.0) -> str:
    """Compute the Sharpe ratio from a series of returns."""
    import json
    returns = json.loads(returns_json)
    # ... computation ...
    return json.dumps({"sharpe_ratio": result})
```

2. Add the tool name to the relevant agent's `build_tools()`:

```python
# pipeline/agents/trainer.py
def build_tools(self, task):
    return build_toolset(
        "create_experiment",
        "validate_experiment",
        "compute_sharpe_ratio",  # new tool
        ...
    )
```

3. Ensure the module is imported somewhere so the `@tool` decorator runs. Typically this happens via the agent module's imports.

The tool registry infers the JSON schema from your function's type hints. No manual schema writing required.

### Adding a New Agent

1. Create a subclass of `BaseAgentWrapper`:

```python
# pipeline/agents/evaluator.py
from __future__ import annotations
from pipeline.agents.base import BaseAgentWrapper
from pipeline.tools.registry import build_toolset
from pipeline.prompts.fragments import assemble_prompt

class EvaluatorAgent(BaseAgentWrapper):
    agent_name = "evaluator"

    def build_system_prompt(self, task: dict) -> str:
        return assemble_prompt("evaluator", task.get("channel", "default"), task)

    def build_tools(self, task: dict):
        return build_toolset(
            "compare_results",
            "session_summary",
            "analyze_experiment_trends",
        )

    def build_context(self, task: dict):
        context = []
        if "comparison" in task:
            context.append({
                "role": "user",
                "content": f"Current rankings:\n{task['comparison']}"
            })
        return context
```

2. Create a prompt module:

```python
# pipeline/prompts/evaluator_prompts.py
from __future__ import annotations
from pipeline.prompts.fragments import register_fragment

register_fragment("evaluator", "default", "role",
    "You are an experiment evaluator...", priority=10)
register_fragment("evaluator", "default", "instructions",
    "Analyze the results and produce a ranking...", priority=20)
```

3. The agent is automatically discoverable via `resolve_agent("evaluator")` through dynamic import.

### Adding a New Backbone Block or Prediction Head

Use the ComponentAuthor agent:

```bash
synth-city agent --name author --message "Create a MambaBlock that uses selective state spaces"
```

Or write components manually. All backbone blocks must conform to the uniform tensor interface:

```python
# Input:  (batch, seq, d_model)
# Output: (batch, seq, d_model)
```

Write the component to `src/models/components/` and call `reload_registry()` to make it available.

### Adding a Pipeline Stage

The PipelineArchitect agent can add stages dynamically, or you can modify the pipeline definition in `pipeline/pipeline_def.py`:

```python
StageSpec(
    name="evaluator",
    agent_name="evaluator",
    position="validate",
    protected=False,
    retry=True,
    optional=True,
    user_message="Evaluate the experiment results.",
)
```

Position options: `"plan"`, `"execute"`, `"validate"`, `"publish"`, `"post"`.

---

## Project Structure

```
synth-city/
├── main.py                          CLI entry point (8+ subcommands)
├── config.py                        Environment-based config
├── pyproject.toml                   Package metadata, deps, tool config
├── requirements.txt                 Direct dependency listing
├── .env.example                     Template environment file
├── Dockerfile                       GPU-capable training image
├── docker-compose.yml               Service definitions
│
├── cli/
│   ├── app.py                       synth-city CLI command
│   ├── display.py                   Rich-powered terminal output
│   ├── dashboard.py                 Dashboard rendering
│   └── score_dashboard.py           Scoring dashboard
│
├── pipeline/
│   ├── orchestrator.py              Agent chaining, retries, stall detection
│   ├── pipeline_def.py              Stage definitions (StageSpec)
│   ├── meta_strategy.py             Adaptive retry/temperature tuning
│   ├── bootstrap.py                 Directory and storage initialization
│   ├── monitor.py                   Event monitoring
│   ├── providers/
│   │   ├── simple_agent.py          Core agent loop (~200 lines)
│   │   └── chutes_client.py         Chutes AI LLM client
│   ├── agents/
│   │   ├── base.py                  BaseAgentWrapper
│   │   ├── planner.py               Experiment planning
│   │   ├── trainer.py               Experiment execution
│   │   ├── code_checker.py          SN50 validation
│   │   ├── debugger.py              Error diagnosis and fixing
│   │   ├── publisher.py             HF Hub + tracking
│   │   ├── author.py                Component authoring
│   │   ├── agent_designer.py        Agent creation
│   │   └── pipeline_architect.py    Pipeline composition
│   ├── tools/
│   │   ├── registry.py              @tool decorator + build_toolset()
│   │   ├── research_tools.py        ResearchSession API
│   │   ├── training_tools.py        Local + Basilica GPU training
│   │   ├── publish_tools.py         HF Hub + W&B
│   │   ├── hippius_store.py         S3-compatible storage
│   │   ├── analysis_tools.py        Experiment analysis
│   │   ├── market_data.py           Pyth Network price feeds
│   │   ├── data_loader.py           Training data management
│   │   ├── file_tools.py            File I/O
│   │   ├── check_shapes.py          SN50 shape validation
│   │   ├── register_tools.py        Component registration
│   │   ├── proxy_tools.py           Low-cost architecture reasoning
│   │   ├── orchestration_tools.py   Pipeline modification
│   │   ├── meta_strategy_tools.py   Strategy tuning
│   │   ├── agent_tools.py           Agent authoring
│   │   └── tool_authoring.py        Tool authoring
│   └── prompts/
│       ├── fragments.py             Composable prompt building blocks
│       ├── planner_prompts.py       Phased reasoning
│       ├── trainer_prompts.py       Experiment execution
│       ├── checker_prompts.py       Validation checklist
│       ├── debugger_prompts.py      Error pattern catalog
│       ├── publisher_prompts.py     Publishing procedure
│       ├── author_prompts.py        Component authoring
│       ├── agent_designer_prompts.py Agent creation
│       └── pipeline_architect_prompts.py Pipeline composition
│
├── compute/
│   ├── basilica.py                  Basilica GPU client (SN39)
│   └── training_server.py           HTTP training server for GPU pods
│
├── subnet/
│   ├── config.py                    SN50 constants
│   ├── miner.py                     Prediction generation + submission
│   ├── validator.py                 CRPS scoring
│   └── score_tracker.py             Local scoring emulator
│
├── integrations/
│   └── openclaw/
│       ├── bridge.py                HTTP bridge server (50+ endpoints)
│       ├── bot_sessions.py          Per-bot session isolation
│       ├── client.py                Python client for bridge
│       ├── publish.py               ClawHub publisher
│       └── skill/                   OpenClaw skill definition
│           ├── SKILL.md             Skill documentation
│           ├── tools.py             Tool definitions
│           └── clawhub.json         Package metadata
│
├── tests/                           Test suite (pytest)
├── docs/                            Documentation
├── workspace/                       Experiment artifacts (auto-created)
└── logs/                            Log files (auto-created)
```

---

## Contributing

### Branch Strategy

Development happens on feature branches. The `main` branch is the stable release branch.

```bash
git checkout -b feature/my-feature
# ... make changes ...
git commit -m "Add new feature"
git push -u origin feature/my-feature
```

### Pull Request Process

1. Ensure all CI checks pass: `ruff check .`, `mypy .`, `pytest`
2. Write a clear PR description explaining what changed and why
3. Keep PRs focused — one feature or fix per PR
4. Add tests for new functionality when practical

### Code Review Checklist

- [ ] Type hints on all new function signatures
- [ ] `from __future__ import annotations` at the top of new files
- [ ] Line length under 100 characters
- [ ] No hard-coded credentials or API keys
- [ ] Tools registered via `@tool` decorator (no manual schema)
- [ ] Agent tools scoped via `build_tools()` (no unnecessary access)

---

## Key Decisions Log

| Decision | Rationale |
|----------|-----------|
| For-loop agent, not DAGs | Readable, debuggable, no framework dependency. See [Philosophy](PHILOSOPHY.md). |
| Decorator-based tool registry | One-line registration, schema inferred from types, no config files. |
| Composition over inheritance | Flat hierarchy, each agent independently declares needs, easy to test. |
| Environment-based config | No code changes between environments, secrets stay out of source control. |
| Fragment-based prompts | Composable, swappable, support variable substitution. Change behavior by editing text. |
| Centralized argument coercion | Handle LLM output sloppiness once in `_coerce_args()`, not in every tool. |
| Per-agent model selection | Use large models where reasoning matters, small models for mechanical tasks. |
| Hippius for persistence | Decentralized, S3-compatible, cross-run learning. Any S3 backend works as fallback. |
