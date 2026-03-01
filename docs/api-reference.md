# API Reference

Complete reference for all public modules, classes, and functions in synth-city. Organized by layer: CLI, pipeline engine, agents, tools, compute, subnet, and integrations.

---

## Table of Contents

- [CLI Entry Points](#cli-entry-points)
- [Pipeline Engine](#pipeline-engine)
  - [PipelineOrchestrator](#pipelineorchestrator)
  - [SimpleAgent](#simpleagent)
  - [Chutes Client](#chutes-client)
- [Agent Framework](#agent-framework)
  - [BaseAgentWrapper](#baseagentwrapper)
  - [PlannerAgent](#planneragent)
  - [TrainerAgent](#traineragent)
  - [CodeCheckerAgent](#codecheckeragent)
  - [DebuggerAgent](#debuggeragent)
  - [PublisherAgent](#publisheragent)
  - [ComponentAuthorAgent](#componentauthoragent)
  - [AgentDesignerAgent](#agentdesigneragent)
  - [PipelineArchitectAgent](#pipelinearchitectagent)
- [Tool Registry](#tool-registry)
- [Tools](#tools)
  - [Research Tools](#research-tools)
  - [Training Tools](#training-tools)
  - [Publishing Tools](#publishing-tools)
  - [Analysis Tools](#analysis-tools)
  - [Hippius Storage Tools](#hippius-storage-tools)
  - [Market Data Tools](#market-data-tools)
  - [File Tools](#file-tools)
  - [Component Registration Tools](#component-registration-tools)
  - [Shape Validation Tools](#shape-validation-tools)
  - [Proxy Tools](#proxy-tools)
  - [Orchestration Tools](#orchestration-tools)
  - [Meta-Strategy Tools](#meta-strategy-tools)
  - [Agent Authoring Tools](#agent-authoring-tools)
  - [Tool Authoring Tools](#tool-authoring-tools)
- [Compute](#compute)
  - [BasilicaGPUClient](#basilicagpuclient)
- [Subnet](#subnet)
  - [SynthMiner](#synthminer)
  - [Validator (CRPS)](#validator-crps)
  - [ScoreTracker](#scoretracker)
- [Integrations](#integrations)
  - [Bridge Server](#bridge-server)
  - [SynthCityClient](#synthcityclient)
  - [Bot Sessions](#bot-sessions)
- [Prompt System](#prompt-system)

---

## CLI Entry Points

**Module:** `main.py`

The CLI is exposed as the `synth-city` command (defined in `pyproject.toml` as `cli.app:main`). All subcommands are also accessible via `python main.py <command>`.

| Command | Function | Description |
|---------|----------|-------------|
| `pipeline` | `cmd_pipeline(args)` | Run full agentic pipeline |
| `sweep` | `cmd_sweep(args)` | Run preset sweep via ResearchSession |
| `experiment` | `cmd_experiment(args)` | Run single experiment |
| `quick` | `cmd_quick(args)` | One-liner convenience experiment |
| `bridge` | `cmd_bridge(args)` | Start HTTP bridge server |
| `client` | `cmd_client(args)` | CLI client for the bridge |
| `history` | `cmd_history(args)` | Query experiment history |
| `score` | `cmd_score(args)` | Start scoring emulator daemon |
| `agent` | `cmd_agent(args)` | Run single agent in isolation |
| `data` | — | Download/inspect training data |

### Pipeline Flags

```
--channel TEXT      Prompt fragment channel (default: "default")
--retries INT       Max retry attempts per stage (default: 5)
--temperature FLOAT Starting LLM temperature (default: 0.1)
--publish           Include publisher stage
```

### Agent Flags

```
--name TEXT         Agent name (required): planner, trainer, codechecker, debugger, publisher, author
--message TEXT      Custom user message (default: "Begin the task.")
--temperature FLOAT LLM temperature (default: 0.1)
```

---

## Pipeline Engine

### PipelineOrchestrator

**Module:** `pipeline/orchestrator.py`

Chains agents into a pipeline with retry logic, temperature escalation, and stall detection.

```python
class PipelineOrchestrator:
    def __init__(
        self,
        max_retries: int = 5,
        base_temperature: float = 0.1,
        temperature_step: float = 0.1,
        publish: bool = False,
    ) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_retries` | `int` | `5` | Maximum retry attempts per pipeline stage |
| `base_temperature` | `float` | `0.1` | Starting LLM temperature |
| `temperature_step` | `float` | `0.1` | Temperature increment per retry |
| `publish` | `bool` | `False` | Whether to include the publisher stage |

#### `run(task) -> dict[str, Any]`

Execute the full pipeline. Returns a result dictionary:

```python
{
    "stages": [...],           # per-stage results
    "success": True,           # overall pass/fail
    "best_experiment": {...},  # winning experiment config
    "best_crps": 0.0123,      # best CRPS score
    "comparison": {...},       # ranked experiment comparison
}
```

#### `_run_with_retry(agent_cls, task, stage_name) -> AgentResult`

Run a single stage with escalating temperature on failure. Builds failure context from prior attempts and detects non-recoverable errors.

#### `_check_debug_loop(task) -> dict[str, Any]`

Run the alternating CodeChecker/Debugger loop with stall detection. Returns:

```python
{"passed": bool, "attempts": int, "stages": [...]}
```

#### `resolve_agent(agent_name) -> type[BaseAgentWrapper] | None`

Static method. Resolves an agent class by name. Checks core agents first, then attempts dynamic import from `pipeline/agents/`.

---

### SimpleAgent

**Module:** `pipeline/providers/simple_agent.py`

The core agent execution loop — a ~200-line for-loop that drives all agent behavior.

#### Data Classes

```python
@dataclass
class ToolResult:
    tool_call_id: str       # ID from the LLM response
    name: str               # tool function name
    content: str            # serialized result
    is_finish: bool = False # True if this was the finish tool
    structured: Any = None  # parsed structured output (from finish)

@dataclass
class AgentResult:
    success: bool                          # did the agent complete successfully
    structured: Any = None                 # structured output from finish tool
    raw_text: str = ""                     # last assistant text response
    messages: list[dict[str, Any]] = ...   # full conversation history
    turns_used: int = 0                    # number of LLM round-trips
```

#### Constructor

```python
class SimpleAgent:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        system_prompt: str,
        tools: dict[str, Callable],
        tool_schemas: list[dict[str, Any]] | None = None,
        max_turns: int = 50,
        temperature: float = 0.1,
    ) -> None
```

#### `run(user_message, context=None) -> AgentResult`

Main execution loop. Sends messages to the LLM, executes tool calls, appends results, and repeats until the agent finishes or `max_turns` is reached.

```python
agent = SimpleAgent(
    client=get_chutes_client(),
    model="Qwen/Qwen3-235B-A22B",
    system_prompt="You are a research planner...",
    tools={"list_blocks": list_blocks_fn},
    tool_schemas=[{"type": "function", "function": {...}}],
)
result = agent.run("Design an experiment plan for SN50.")
```

#### `inject_message(messages, role, content) -> None`

Inject a message into the conversation mid-run. Used by the orchestrator to inject stall warnings.

#### `_coerce_args(params, schema) -> dict[str, Any]`

Static function. Normalizes LLM-produced arguments against the expected schema. Handles empty strings, JSON-in-strings, string booleans, and type mismatches.

---

### Chutes Client

**Module:** `pipeline/providers/chutes_client.py`

#### `get_chutes_client() -> OpenAI`

Returns a cached singleton OpenAI client configured for the Chutes AI endpoint. Timeout: 120s connect, 300s read, 60s write. Cached via `@lru_cache(maxsize=1)`.

#### `chat_completion_with_backoff(client, *, max_retries=4, base_delay=2.0, **kwargs) -> ChatCompletion`

Wraps the OpenAI SDK's `chat.completions.create()` with exponential backoff on transient failures. Delays: 2s, 4s, 8s, 16s. Raises `RuntimeError` after all retries exhausted.

---

## Agent Framework

### BaseAgentWrapper

**Module:** `pipeline/agents/base.py`

Abstract base class for all agents. Provides the `run()` method that builds a `SimpleAgent` and executes it.

```python
class BaseAgentWrapper:
    agent_name: str = "base"  # override in subclasses

    def __init__(
        self,
        client: OpenAI | None = None,
        model: str | None = None,
        max_turns: int | None = None,
        temperature: float = 0.1,
    ) -> None
```

| Method | Required | Description |
|--------|----------|-------------|
| `build_system_prompt(task)` | Yes | Return the system prompt string |
| `build_tools(task)` | Yes | Return `(tools_dict, schemas_list)` |
| `build_context(task)` | No | Return list of context messages to inject |
| `post_process(result, task)` | No | Transform the agent's output |
| `run(task)` | Inherited | Build SimpleAgent and execute |

### PlannerAgent

**Module:** `pipeline/agents/planner.py`
**Agent name:** `"planner"`
**Default model:** `Qwen/Qwen3-235B-A22B`

Surveys available components and past experiment history, then produces a structured experiment plan. Uses two-phase prompting: diagnostic first, then execution plan.

**Tools:** `list_blocks`, `list_heads`, `list_presets`, `session_summary`, `compare_results`, `load_hippius_history`, `load_hippius_run`, `fetch_experiment_runs`, `analyze_experiment_trends`, `list_hf_models`, `scan_experiment_history`, `check_experiment_novelty`, `estimate_params`, `estimate_flops`, `generate_ablation_configs`, `sweep_configs`

**Context injection:** CRPS scores and prior comparison data (if available).

### TrainerAgent

**Module:** `pipeline/agents/trainer.py`
**Agent name:** `"trainer"`
**Default model:** `Qwen/Qwen3-235B-A22B`

Executes the Planner's experiment plan via ResearchSession. Trains on Basilica GPU deployments.

**Tools:** `create_experiment`, `validate_experiment`, `compare_results`, `session_summary`, `flush_session`, `data_loader_info`, `load_hippius_history`, `fetch_experiment_runs`, `scan_experiment_history`, `check_experiment_novelty`, `estimate_params`, `estimate_flops`, `generate_ablation_configs`, `sweep_configs`, `probe_architecture`, `probe_batch`, `check_gpu_balance`, `create_training_deployment`, `get_training_deployment`, `get_deployment_logs`, `list_deployments`, `delete_training_deployment`, `wait_for_deployment_ready`, `run_experiment_on_deployment`

**Context injection:** Planner output via `task["plan"]`.

### CodeCheckerAgent

**Module:** `pipeline/agents/code_checker.py`
**Agent name:** `"codechecker"`
**Default model:** `Qwen/Qwen2.5-Coder-32B-Instruct`

Validates experiment configs against SN50 specifications. Must call `validate_experiment()` before finishing.

**Tools:** `validate_experiment`, `describe_experiment`, `list_blocks`, `list_heads`

**Context injection:** Experiment config and run result.

### DebuggerAgent

**Module:** `pipeline/agents/debugger.py`
**Agent name:** `"debugger"`
**Default model:** `Qwen/Qwen2.5-Coder-32B-Instruct`

Diagnoses and fixes failed experiments. Receives error reports and failed configs as context. Has access to GPU tools for re-running fixed experiments.

**Tools:** `create_experiment`, `validate_experiment`, `list_blocks`, `list_heads`, `load_hippius_history`, `scan_experiment_history`, GPU deployment tools

**Context injection:** Error report and failed experiment config.

### PublisherAgent

**Module:** `pipeline/agents/publisher.py`
**Agent name:** `"publisher"`
**Default model:** `Qwen/Qwen2.5-Coder-32B-Instruct`

Publishes validated models to HuggingFace Hub and logs metrics to tracking services.

**Tools:** `validate_experiment`, `publish_model`, `log_to_trackio`, `save_to_hippius`, `fetch_experiment_runs`, `list_hf_models`, `fetch_hf_model_card`

**Context injection:** Best experiment config, metrics, and full comparison.

### ComponentAuthorAgent

**Module:** `pipeline/agents/author.py`
**Agent name:** `"author"`
**Default model:** `Qwen/Qwen3-235B-A22B`

Writes new backbone blocks and prediction heads into the open-synth-miner component registry.

**Tools:** `list_blocks`, `list_heads`, `list_component_files`, `read_component`, `write_component`, `write_config`, `reload_registry`

**Context injection:** Component spec and reference blocks.

### AgentDesignerAgent

**Module:** `pipeline/agents/agent_designer.py`
**Agent name:** `"agent_designer"`

Creates new pipeline agents, prompt modules, and tools dynamically.

**Tools:** Agent discovery, writing, prompt authoring, tool authoring

### PipelineArchitectAgent

**Module:** `pipeline/agents/pipeline_architect.py`
**Agent name:** `"pipeline_architect"`

Modifies pipeline composition and tunes the orchestrator's meta-strategy.

**Tools:** `get_pipeline`, `add_pipeline_stage`, `remove_pipeline_stage`, `reorder_pipeline_stage`, `get_meta_strategy`, `update_meta_strategy`, `get_run_history`, `analyze_strategy_effectiveness`, `reset_meta_strategy`

---

## Tool Registry

**Module:** `pipeline/tools/registry.py`

### ToolDef

```python
@dataclass
class ToolDef:
    name: str                          # tool name (used in agent tool lists)
    description: str                   # human-readable description
    func: Callable[..., Any]           # the actual function
    parameters_schema: dict[str, Any]  # JSON schema for parameters
```

### `@tool` Decorator

```python
@tool(
    name: str | None = None,          # override function name
    description: str | None = None,   # override docstring
    parameters_schema: dict | None = None,  # override inferred schema
)
def my_function(param: str) -> str:
    ...
```

If `name` is not provided, the function name is used. If `description` is not provided, the docstring is used. If `parameters_schema` is not provided, the schema is inferred from type hints.

### Registry Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_tool` | `(name: str) -> ToolDef \| None` | Look up a single tool by name |
| `get_tools` | `(*names: str) -> dict[str, Callable]` | Get name-to-callable mapping |
| `get_schemas` | `(*names: str) -> list[dict]` | Get OpenAI-format tool schemas |
| `all_tool_names` | `() -> list[str]` | List all registered tool names |
| `build_toolset` | `(*names: str) -> tuple[dict, list]` | Convenience: returns `(tools_dict, schemas_list)` |
| `register_tool_def` | `(tool_def: ToolDef) -> None` | Programmatically register a tool |
| `unregister_tool` | `(name: str) -> None` | Remove a tool from the registry |

### Example

```python
from pipeline.tools.registry import tool, build_toolset

@tool(description="Fetch the latest price for an asset")
def get_price(asset: str) -> str:
    """Returns JSON with price, confidence, and timestamp."""
    ...

# In an agent's build_tools():
tools, schemas = build_toolset("get_price", "list_blocks", "compare_results")
```

---

## Tools

### Research Tools

**Module:** `pipeline/tools/research_tools.py`

The primary interface for creating, running, and comparing experiments. All functions work through the `ResearchSession` API from open-synth-miner.

#### `list_blocks() -> str`

List all available backbone blocks with cost ratings and use-case descriptions. Falls back to a static list if open-synth-miner is not installed.

#### `list_heads() -> str`

List all available prediction head types with expressiveness levels.

#### `list_presets() -> str`

List pre-built block+head combinations ready to run.

#### `create_experiment(...) -> str`

Build an experiment config from components and hyperparameters.

```python
@tool
def create_experiment(
    blocks: str,               # comma-separated block names
    head: str = "GBMHead",     # prediction head
    timeframe: str = "",       # "5m" or "1m" (applies preset defaults)
    d_model: int = 32,         # hidden dimension
    horizon: int = 12,         # prediction steps
    n_paths: int = 100,        # Monte Carlo paths
    lr: float = 0.001,         # learning rate
    seq_len: int = 32,         # input sequence length
    batch_size: int = 4,       # training batch size
    feature_dim: int = 4,      # input features (OHLC)
    head_kwargs: str = "",     # JSON string of head parameters
    block_kwargs: str = "",    # JSON string of block parameters
) -> str
```

Returns a JSON experiment config dict.

#### `validate_experiment(experiment: str) -> str`

Validate an experiment config without running it. Returns parameter count, errors, and warnings.

#### `describe_experiment(experiment: str) -> str`

Full description: blocks, head, parameter count, training configuration.

#### `run_experiment(experiment: str, epochs: int = 1, name: str = "") -> str`

Train a model and return metrics. Auto-saves results to Hippius. Auto-flushes if the session exceeds 100 experiments.

Returns:
```json
{
    "status": "ok",
    "metrics": {"crps": 0.0123, "sharpness": 15.45, "log_likelihood": -2.34},
    "param_count": 45320,
    "training_time_s": 123.45
}
```

#### `compare_results() -> str`

Rank all session experiments by CRPS (best first).

#### `session_summary() -> str`

Summary of the current session: number of experiments, comparison, and all results.

#### `flush_session(keep_top_n: int = 10) -> str`

Save all results to Hippius, clear session memory, and keep only the top N results.

#### `clear_session() -> str`

Reset the research session completely.

---

### Training Tools

**Module:** `pipeline/tools/training_tools.py`

GPU cloud management for training experiments on Basilica (SN39).

#### `check_gpu_balance() -> str`

Check Basilica account balance.

#### `create_training_deployment() -> str`

Spin up a Docker-based GPU pod on Basilica. Uses the image from `BASILICA_DEPLOY_IMAGE`. Includes retry logic (3 attempts with exponential backoff) and optional health checks.

#### `get_training_deployment(name: str) -> str`

Get deployment status including phase (Pending/Running/Failed) and URL.

#### `get_deployment_logs(name: str) -> str`

Fetch container logs from a deployment.

#### `list_deployments() -> str`

List all active GPU deployments.

#### `delete_training_deployment(name: str) -> str`

Terminate a deployment and free GPU resources.

#### `wait_for_deployment_ready(name: str, timeout: int = 300) -> str`

Poll deployment status until it reaches Running phase or timeout.

#### `run_experiment_on_deployment(url: str, experiment: str) -> str`

Send an experiment config to a running GPU deployment for training via HTTP. Returns metrics.

#### `run_training_local(script_path: str, args: str = "") -> str`

Execute a Python training script in a subprocess (600s timeout).

#### `run_python(code: str) -> str`

Execute a Python code snippet (120s timeout).

---

### Publishing Tools

**Module:** `pipeline/tools/publish_tools.py`

#### `publish_model(experiment: str, crps_score: float, repo_id: str = "") -> str`

Publish a model to HuggingFace Hub. Recreates the model from config, computes a recipe hash, and logs to W&B. Returns HF link and shareable report.

#### `log_to_trackio(experiment_name: str, metrics: str, config: str = "") -> str`

Log experiment metrics to Trackio. Persists to Hippius for cross-session history.

---

### Analysis Tools

**Module:** `pipeline/tools/analysis_tools.py`

#### `fetch_experiment_runs(limit: int = 20, order: str = "best") -> str`

Fetch experiment runs from Hippius. `order` options: `"best"` (lowest CRPS), `"recent"`, `"worst"`.

#### `get_experiment_run_detail(run_id: str) -> str`

Full details for a specific run. Pass `"latest"` for the most recent.

#### `analyze_experiment_trends(limit: int = 50) -> str`

CRPS improvement trends over time. Returns timeline, running best CRPS, and improvement trajectory.

#### `list_hf_models(repo_id: str = "") -> str`

List published models on HuggingFace Hub including versions, tags, and download counts.

#### `fetch_hf_model_card(repo_id: str = "", model_id: str = "") -> str`

Fetch a model card/README from HuggingFace Hub.

#### `scan_experiment_history() -> str`

Analyze experiment history for lessons learned and architecture deduplication via fingerprinting.

#### `check_experiment_novelty(...) -> str`

Check if a proposed experiment is novel relative to past runs.

---

### Hippius Storage Tools

**Module:** `pipeline/tools/hippius_store.py`

S3-compatible decentralized storage with lazy-loaded boto3 client and retry logic (3 retries, exponential backoff).

#### `save_to_hippius(experiment: str, result: str, name: str = "") -> str`

Save experiment + result to Hippius storage.

#### `load_hippius_history(limit: int = 20) -> str`

Load experiment history sorted by CRPS.

#### `load_hippius_run(run_id: str) -> str`

Load a specific pipeline run. Pass `"latest"` for the most recent.

#### Internal Functions

| Function | Description |
|----------|-------------|
| `get_run_id() -> str` | Get current run ID (per-bot or global) |
| `reset_run_id() -> None` | Reset for next pipeline run |
| `save_experiment_result(name, experiment, result) -> str \| None` | Persist single experiment |
| `save_pipeline_summary(summary) -> str \| None` | Persist full run summary |
| `save_comparison(comparison) -> str \| None` | Persist CRPS ranking |

---

### Market Data Tools

**Module:** `pipeline/tools/market_data.py`

#### `get_latest_price(asset: str) -> str`

Fetch latest price from the Pyth Network oracle. Returns price, confidence interval, and timestamp.

Supported assets via Pyth feed IDs: BTC, ETH, SOL, XAU. Tokenized assets (SPYX, NVDAX, etc.) fall back to synthetic data.

#### `get_historical_data(asset: str, days: int = 30) -> str`

Fetch OHLCV history. Uses CoinGecko for crypto assets (BTC, ETH, SOL).

#### `compute_returns_stats(price_data_json: str) -> str`

Compute log returns statistics: mean, std, annualized volatility, skewness, kurtosis.

---

### File Tools

**Module:** `pipeline/tools/file_tools.py`

Thread-safe file I/O with per-file write locks and workspace sandboxing.

#### `write_file(path: str, content: str) -> str`

Write content to a file relative to the workspace. Per-bot workspace isolation when running in multi-bot mode.

#### `read_file(path: str) -> str`

Read a file (truncates to 30,000 characters).

#### `list_files(path: str = ".") -> str`

List files in a directory.

#### `delete_file(path: str) -> str`

Delete a file.

#### `append_file(path: str, content: str) -> str`

Append content to a file (thread-safe).

---

### Component Registration Tools

**Module:** `pipeline/tools/register_tools.py`

Thread-safe registry access via `_registry_lock`.

#### `write_component(filename: str, code: str) -> str`

Write a PyTorch block/head to `src/models/components/`. Auto-discovered by the registry.

#### `read_component(path: str) -> str`

Read component source code for reference.

#### `list_component_files() -> str`

List all registered component files.

#### `reload_registry() -> str`

Trigger re-discovery so new components appear immediately.

#### `write_config(filename: str, content: str) -> str`

Write a YAML model config to `configs/model/`.

---

### Shape Validation Tools

**Module:** `pipeline/tools/check_shapes.py`

#### `check_shapes(model_path: str) -> str`

Validate model output shapes for SN50 compliance. Runs in a subprocess with 120s timeout. Checks:

- Shape: `[1000, 289]` (SN50_NUM_PATHS x expected steps)
- No NaN or Inf values
- All values positive (they represent prices)

---

### Proxy Tools

**Module:** `pipeline/tools/proxy_tools.py`

Low-cost architecture reasoning without GPU. Used by Planner and Trainer for experiment design.

#### `estimate_params(blocks: str, head: str, d_model: int) -> str`

Estimate parameter count for a given architecture.

#### `estimate_flops(blocks: str, head: str, d_model: int, seq_len: int, batch_size: int) -> str`

Estimate FLOPs for forward pass.

#### `generate_ablation_configs(base_experiment: str) -> str`

Generate ablation experiment variants from a base config.

#### `sweep_configs(base_experiment: str, param_ranges: str) -> str`

Generate hyperparameter sweep configs.

#### `probe_architecture(experiment: str) -> str`

Quick architecture analysis without training.

#### `probe_batch(batch_size: int, num_assets: int) -> str`

Batch size and memory analysis.

---

### Orchestration Tools

**Module:** `pipeline/tools/orchestration_tools.py`

#### `get_pipeline() -> str`

Return the current pipeline definition as JSON.

#### `add_pipeline_stage(name: str, agent_name: str, position: str, retry: bool = True, optional: bool = False) -> str`

Add a new stage to the pipeline. Protected stages (planner, trainer, check_debug) cannot be removed.

#### `remove_pipeline_stage(name: str) -> str`

Remove a stage (except protected ones).

#### `reorder_pipeline_stage(name: str, new_position: str) -> str`

Change a stage's position in the pipeline sequence.

---

### Meta-Strategy Tools

**Module:** `pipeline/tools/meta_strategy_tools.py`

#### `get_meta_strategy() -> str`

Get current orchestrator strategy parameters.

#### `update_meta_strategy(max_retries=None, base_temperature=None, temperature_step=None, stall_threshold=None) -> str`

Update strategy parameters with bounds checking.

#### `get_run_history(limit: int = 20) -> str`

Fetch run history for analysis.

#### `analyze_strategy_effectiveness() -> str`

Analyze success rates and failure patterns across past runs.

#### `reset_meta_strategy() -> str`

Revert all strategy parameters to defaults.

---

### Agent Authoring Tools

**Module:** `pipeline/tools/agent_tools.py`

#### `list_agents() -> str`

List all agent modules in `pipeline/agents/`.

#### `read_agent(name: str) -> str`

Read agent source code.

#### `write_agent(name: str, content: str) -> str`

Write a new agent module.

#### `list_agent_prompts() -> str`

List prompt modules in `pipeline/prompts/`.

#### `read_agent_prompt(name: str) -> str`

Read prompt module source.

#### `write_agent_prompt(name: str, content: str) -> str`

Write a new prompt module.

---

### Tool Authoring Tools

**Module:** `pipeline/tools/tool_authoring.py`

#### `list_available_tools() -> str`

List all tools in the registry with descriptions.

#### `list_tool_files() -> str`

List tool source files.

#### `read_tool(path: str) -> str`

Read tool source code.

#### `describe_tool(name: str) -> str`

Get tool schema and description.

#### `write_tool(filename: str, content: str) -> str`

Write a new tool module.

#### `validate_tool(name: str) -> str`

Validate a tool's schema.

#### `reload_tools() -> str`

Reload the tool registry.

---

## Compute

### BasilicaGPUClient

**Module:** `compute/basilica.py`

Budget-aware wrapper for the Basilica secure-cloud GPU marketplace (Bittensor SN39).

```python
class BasilicaGPUClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_hourly_rate: float | None = None,
        allowed_gpu_types: list[str] | None = None,
    ) -> None
```

| Method | Returns | Description |
|--------|---------|-------------|
| `list_cheap_gpus()` | `list[GpuOffering]` | Offerings filtered by budget and GPU type |
| `rent_gpu(offering_id, ssh_key_id)` | `SecureCloudRentalResponse` | Rent a specific GPU |
| `rent_cheapest(ssh_key_id)` | `SecureCloudRentalResponse` | Rent the cheapest available GPU |
| `stop_rental(rental_id)` | `dict` | Stop rental, returns cost summary |
| `list_active_rentals()` | `list[dict]` | All active rentals |
| `get_rental_status(rental_id)` | `dict` | Detailed status with SSH info |
| `get_balance()` | `dict` | Account balance |
| `create_deployment(...)` | `DeploymentResponse` | Create Docker-based GPU pod |
| `get_deployment(name)` | `DeploymentResponse` | Deployment status |
| `get_deployment_logs(name, tail=100)` | `str` | Container logs |
| `delete_deployment(name)` | `dict` | Terminate deployment |
| `list_deployments()` | `list[dict]` | All deployments |
| `ensure_ssh_key(name, public_key_path)` | `str` | Ensure SSH key is registered |

#### Deployment Example

```python
from compute.basilica import BasilicaGPUClient

client = BasilicaGPUClient()
deployment = client.create_deployment(
    name="training-run-001",
    image="ghcr.io/tensorlink-ai/synth-city-gpu:latest",
    gpu_model="RTX-A4000",
)
print(f"URL: {deployment.url}, Phase: {deployment.phase}")

# Wait for pod to be ready, then train
logs = client.get_deployment_logs("training-run-001")

# Clean up
client.delete_deployment("training-run-001")
```

---

## Subnet

### SynthMiner

**Module:** `subnet/miner.py`

Generates and submits probabilistic price forecasts for SN50.

```python
class SynthMiner:
    def register_model(self, asset: str, model: Forecaster) -> None
    def set_price(self, asset: str, price: float) -> None
    def generate_prediction(self, asset: str, horizon: str = "24h") -> dict
    def generate_all_predictions(self, horizon: str = "24h") -> list[dict]
    def format_submission(self, predictions: list[dict]) -> dict
```

The `Forecaster` protocol requires:

```python
def generate_paths(
    asset: str,
    num_paths: int,    # 1000
    num_steps: int,    # 289 for 24h
    s0: float | None,  # starting price
) -> np.ndarray       # shape: (num_paths, num_steps)
```

#### Prediction Validation

`generate_prediction()` validates output shape `(SN50_NUM_PATHS, num_steps)` and checks for NaN, Inf, and non-positive values before returning the payload.

---

### Validator (CRPS)

**Module:** `subnet/validator.py`

CRPS scoring implementation for local evaluation.

#### `crps_ensemble(forecasts: np.ndarray, observation: float) -> float`

Compute CRPS for an ensemble of forecasts at a single timestep:

```
CRPS = mean(|forecast_i - observation|) - 0.5 * mean(|forecast_i - forecast_j|)
```

#### `crps_basis_points(forecasts: np.ndarray, observation: float, reference_price: float) -> float`

CRPS normalized in basis points (0.01% = 1 bp) to handle different asset price scales.

#### `evaluate_prediction(paths, realized_prices, step_minutes) -> dict[str, float]`

Evaluate a full prediction against realized prices. Returns CRPS at 9 time horizons plus final and sum scores.

#### `evaluate_multi_asset(predictions, realized) -> dict[str, Any]`

Per-asset CRPS with weighting from `SN50_ASSETS`. Returns per-asset scores, weighted CRPS sum, and counts.

---

### ScoreTracker

**Module:** `subnet/score_tracker.py`

Local replica of the SN50 validator scoring loop. Thread-safe.

```python
class ScoreTracker:
    def record_prompt(self, ...) -> None
    def score_prompt_with_realized(self, ...) -> None
    def collect_price_snapshot(self, ...) -> None
    def score_prompt_deferred(self, ...) -> None
    def score_pending(self) -> None
    def save_daily_summary(self, date_str: str) -> None
    def save_leaderboard(self) -> None
    def get_status(self) -> dict
    def get_recent_scores(self) -> list
    def load_prompt_history(self) -> list
    def load_leaderboard(self) -> dict
```

#### ScoringDaemon

Background thread for continuous prompt generation and scoring.

```python
class ScoringDaemon:
    def __init__(self, tracker: ScoreTracker, interval_minutes: int = 30)
    def start(self) -> None
    def stop(self) -> None
    @property
    def running(self) -> bool
```

---

## Integrations

### Bridge Server

**Module:** `integrations/openclaw/bridge.py`

HTTP bridge exposing synth-city operations as a REST API. Supports multi-bot concurrency with per-bot session isolation.

#### Authentication

Optional API key authentication via `X-API-Key` header (enabled when `BRIDGE_API_KEY` is set). Uses HMAC-safe comparison.

#### Bot Identification

The `X-Bot-Id` header identifies the requesting bot. Each bot gets an isolated session. Without the header, the "default" session is used.

#### Endpoints Reference

**Health and Admin:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check, active bot count |
| GET | `/bots/sessions` | List all active bot sessions |
| GET | `/bots/session/:id` | Get specific bot session |
| DELETE | `/bots/session/:id` | Remove a bot session |

**Pipeline:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/pipeline/run` | Start full pipeline (async, per-bot) |
| GET | `/pipeline/status` | Current pipeline status |

**Experiments:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/experiment/create` | Create experiment config |
| POST | `/experiment/run` | Run experiment, return metrics |
| POST | `/experiment/validate` | Validate config without running |
| GET | `/experiment/compare` | Session results ranked by CRPS |
| GET | `/experiment/compare/all` | Merged ranking across all bots |

**Components:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/components/blocks` | List backbone blocks |
| GET | `/components/heads` | List head types |
| GET | `/components/presets` | List presets |

**Session:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/session/summary` | Research session summary |
| POST | `/session/clear` | Reset session |

**Market Data:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/market/price/:asset` | Live Pyth oracle price |
| GET | `/market/history/:asset` | Historical OHLCV data |

**Registry:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/registry/files` | List component files |
| GET | `/registry/read` | Read component source |
| POST | `/registry/write` | Write new block/head |
| POST | `/registry/reload` | Reload registry |

**HuggingFace Hub:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/hf/models` | List published models |
| GET | `/hf/model-card` | Fetch model card |
| GET | `/hf/artifact` | Download JSON artifact |

**History:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/history/runs` | List pipeline runs from Hippius |
| GET | `/history/run/:run_id` | Load specific run |
| GET | `/history/experiments` | Best experiments across runs |
| GET | `/history/trackio` | Experiment runs with filtering |

**Agent Design:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/agents/list` | List agents |
| GET | `/agents/read` | Read agent source |
| POST | `/agents/write` | Write new agent |
| GET | `/agents/prompts/list` | List prompt modules |
| GET | `/agents/prompts/read` | Read prompt source |
| POST | `/agents/prompts/write` | Write new prompt |
| GET | `/agents/tools` | List registered tools |

**Dashboard:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/dashboard` | Serve HTML dashboard |
| GET | `/dash/snapshot` | Monitoring snapshot |
| GET | `/dash/events` | Event stream |

---

### SynthCityClient

**Module:** `integrations/openclaw/client.py`

Python HTTP client for the bridge server.

```python
class SynthCityClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8377",
        timeout: float = 300,
        retries: int = 3,
        api_key: str = "",
        bot_id: str = "",
    ) -> None
```

Retry logic: exponential backoff on transient errors (ConnectError, ConnectTimeout, ReadTimeout). Does not retry HTTP 4xx/5xx errors.

Methods mirror all bridge endpoints. See bridge endpoints table above for the full list.

#### Exception

```python
class BridgeConnectionError(Exception):
    """Raised when the bridge server is unreachable."""
```

---

### Bot Sessions

**Module:** `integrations/openclaw/bot_sessions.py`

#### BotSession

```python
@dataclass
class BotSession:
    bot_id: str
    pipeline_state: PipelineState
    run_id: str                    # format: YYYYMMDD-HHMMSS[-bot_id]-{uuid8}
    workspace_dir: Path
    last_active: float
```

| Method | Description |
|--------|-------------|
| `touch()` | Update last-active timestamp |
| `acquire_request()` | Increment active request counter |
| `release_request()` | Decrement active request counter |
| `get_research_session()` | Return lazily-initialized ResearchSession |
| `reset_run_id()` | Generate new run ID for next pipeline run |

#### SessionRegistry

Thread-safe registry with TTL-based cleanup.

```python
class SessionRegistry:
    def get_or_create(self, bot_id: str) -> BotSession
    def get(self, bot_id: str) -> BotSession | None
    def remove(self, bot_id: str) -> bool
    def list_sessions(self) -> list[dict]
    def active_count(self) -> int
```

Background reaper thread evicts idle sessions after `BOT_SESSION_TTL_SECONDS` (default: 3600). The "default" session and sessions with active pipelines or in-flight requests are never evicted.

#### Context Functions

```python
def get_current_session() -> BotSession | None    # returns None in CLI mode
def set_current_session(session) -> Token
def reset_current_session(token) -> None
```

---

## Prompt System

**Module:** `pipeline/prompts/fragments.py`

### Fragment

```python
@dataclass
class Fragment:
    key: str        # unique name within agent+channel
    content: str    # prompt text (supports {variable} substitution)
    priority: int   # lower priority = earlier in assembled prompt
```

### Functions

#### `register_fragment(agent_name, channel, key, content, priority) -> None`

Register a prompt fragment for a specific agent and channel combination.

#### `get_fragments(agent_name, channel) -> list[Fragment]`

Retrieve all fragments for an agent+channel, sorted by priority.

#### `assemble_prompt(agent_name, channel, task) -> str`

Assemble the full system prompt by concatenating fragments (sorted by priority) with variable substitution from the task dict.

### Per-Agent Prompt Modules

Each agent has a dedicated prompt module that registers its fragments on import:

| Module | Agent | Strategy |
|--------|-------|----------|
| `planner_prompts.py` | Planner | Two-phase reasoning (diagnostic + plan) |
| `trainer_prompts.py` | Trainer | Experiment execution protocol |
| `checker_prompts.py` | CodeChecker | Structured validation checklist |
| `debugger_prompts.py` | Debugger | Error pattern catalog with known fixes |
| `publisher_prompts.py` | Publisher | Publishing procedure with validation gate |
| `author_prompts.py` | Author | Component authoring with tensor interface contract |
| `agent_designer_prompts.py` | AgentDesigner | Agent creation guidelines |
| `pipeline_architect_prompts.py` | PipelineArchitect | Pipeline composition reasoning |
