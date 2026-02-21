# Building a Self-Extending AI Research Pipeline

*How synth-city chains specialized agents to autonomously discover, train, and publish probabilistic forecasting models — and how it extends its own capabilities at runtime.*

---

Most AI pipelines are closed systems. You define the tools, wire up the agents, and press run. The system does exactly what it was configured to do at design time — no more. That's fine for stable, well-understood tasks. It's a poor fit for open-ended research, where the right tools, strategies, and model architectures are precisely what you're trying to *discover*.

synth-city takes a different approach. It's an autonomous research pipeline for probabilistic price forecasting on [Bittensor Subnet 50 (Synth)](https://github.com/tensorlink-dev/open-synth-miner) — a competition where miners produce 1,000 Monte Carlo price paths per asset at 5-minute intervals over 24 hours, scored by CRPS (Continuous Ranked Probability Score, lower is better). The pipeline wraps [open-synth-miner](https://github.com/tensorlink-dev/open-synth-miner), a composable PyTorch framework with 15 backbone blocks, 6 prediction heads, and 10 presets, and chains specialized agents together to search that space automatically.

The interesting part isn't the forecasting task — it's the architecture. synth-city exposes structured mechanisms by which the system can extend itself: growing its model component vocabulary at runtime and, more ambitiously, authoring new reasoning agents. The live market score acts as an external anchor that keeps self-extension grounded in reality.

This post walks through the architecture layer by layer, with an honest accounting of what works, what doesn't, and what this pattern might actually achieve.

---

## The Three Layers

The architecture separates capability into three tiers with sharply different properties. Getting this separation right is what makes the whole thing tractable.

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                         │
│   Sequences agents, retries, escalates, detects stalls  │
├───────────────┬───────────────┬────────────────────────┤
│    Planner    │    Trainer    │  CodeChecker / Debugger │   ← AGENTS
│    Author     │   Publisher   │       AgentDesigner     │
├───────────────┴───────────────┴────────────────────────┤
│                      TOOLS                              │
│  @tool-decorated functions — stateless, atomic, typed   │
│  list_blocks · run_experiment · write_file · fetch_prices│
└─────────────────────────────────────────────────────────┘
```

**Layer 1: Tools** are stateless atomic operations — read a file, call an API, run an experiment, fetch price data. Each is a plain Python function decorated with `@tool`. No memory, no reasoning, no hidden state. The global tool registry is the system's capability inventory.

**Layer 2: Agents** are reasoning loops. Each agent receives a system prompt (assembled from composable fragments), a *scoped subset* of the tool registry, and an LLM. The loop is intentionally simple — send message, receive tool calls, execute them, append results, repeat. All the intelligence lives in the prompts and the LLM, not in the framework scaffolding.

**Layer 3: The Orchestrator** sequences agents into a pipeline with retry logic, temperature escalation, and stall detection. It's the only component with a global view of pipeline state. It's also the only layer that isn't extensible — a deliberate choice we'll come back to.

---

## The Agent Loop: A For-Loop, Not a DAG

The most important design decision in synth-city is what *isn't* there. No LangGraph. No AutoGPT. No task queues or planning graphs. The core agent loop is ~200 lines of straightforward Python:

```python
for turn in range(max_turns):
    response = llm.chat(messages, tools)

    if not response.tool_calls:
        return AgentResult(success=True, ...)  # implicit finish

    for tc in response.tool_calls:
        result = execute_tool(tc)
        messages.append(result)
        if result.is_finish:
            return AgentResult(...)           # explicit finish

return AgentResult(success=False, raw_text="max turns exhausted")
```

That's it. The LLM *is* the planner. Prompts guide the reasoning. The loop just keeps the conversation going until the agent signals it's done — either by calling the built-in `finish` tool explicitly, or by responding with no tool calls at all (the implicit path for models that narrate their conclusions).

Two things make this robust in practice:

**Argument coercion.** LLMs produce imperfect JSON. A `list` parameter might arrive as `"[]"` (a string), or `""` (empty string), or `[[...]]` (double-nested). A `bool` might arrive as `"true"` instead of `true`. Rather than making every tool handle these edge cases, `_coerce_args()` normalises all arguments before dispatch — once, in one place. Tool authors write clean Python with real types and never see the mess.

**Provider independence.** The LLM client speaks the OpenAI-compatible API. Swapping providers means changing one URL and one API key. The agent loop is completely provider-agnostic.

---

## The Tool Registry: One Decorator, No Config

Adding a capability to an agent should be a one-liner. In synth-city, it is:

```python
@tool(description="Fetch the latest price data for an asset")
def fetch_prices(asset: str, interval: str = "5m") -> str:
    ...
```

The `@tool` decorator registers the function in a global dictionary, infers a JSON schema from the Python type hints (`str → "string"`, `int → "integer"`, `list → "array"`), and marks parameters without defaults as required. No config files. No factory classes. No registration calls. The tool exists the moment the module is imported.

The schema inference happens at decoration time:

```
Python signature:  fetch_prices(asset: str, interval: str = "5m")
                                    ↓
JSON schema:       { "asset": {"type": "string"},      ← required
                    "interval": {"type": "string"} }   ← optional (has default)
```

### Dynamic Scoping

Not every agent should see every tool. A Planner that can accidentally start training is a liability. A Publisher that can create new experiments is a bug waiting to happen.

Each agent's `build_tools()` method returns a list of tool names, and `build_toolset()` pulls only those from the global registry:

```
Global Registry                    Per-Agent Scopes
┌──────────────────────┐           ┌─────────────────────┐
│ list_blocks          │──────────▶│ PLANNER             │
│ list_heads           │──────────▶│ list_blocks         │
│ list_presets         │──────────▶│ list_heads          │
│ create_experiment    │           │ list_presets        │
│ run_experiment       │           │ compare_results     │
│ validate_experiment  │           └─────────────────────┘
│ compare_results      │           ┌─────────────────────┐
│ write_file           │──────────▶│ TRAINER             │
│ read_file            │──────────▶│ create_experiment   │
│ fetch_prices         │──────────▶│ run_experiment      │
│ publish_to_hub       │──────────▶│ validate_experiment │
│ log_to_wandb         │           │ compare_results     │
│ ...                  │           └─────────────────────┘
└──────────────────────┘           ┌─────────────────────┐
                                   │ PUBLISHER           │
                                   │ publish_to_hub      │
                                   │ log_to_wandb        │
                                   └─────────────────────┘
```

Each agent operates in a constrained action space. Separation of concerns is enforced structurally, not by convention.

---

## Agents: Thin Wrappers, Not Deep Hierarchies

Every agent subclasses `BaseAgentWrapper` and overrides exactly four hooks:

| Hook | Purpose |
|---|---|
| `build_system_prompt(task)` | Assemble personality and instructions |
| `build_tools(task)` | Select which tools this agent can use |
| `build_context(task)` | Inject prior conversation or data |
| `post_process(result, task)` | Clean up or transform output |

A complete agent implementation looks like this:

```python
class PlannerAgent(BaseAgentWrapper):
    agent_name = "planner"

    def build_system_prompt(self, task):
        return assemble_prompt("planner", task.get("channel"), task)

    def build_tools(self, task):
        return build_toolset("list_blocks", "list_heads", "list_presets",
                             "compare_results", "fetch_prices", ...)

    def build_context(self, task):
        context = []
        if "crps_scores" in task:
            context.append({"role": "user", "content": format_scores(task)})
        return context
```

One class, one file, no surprises. There's no `super().build_tools()` to worry about, no shared base state that siblings can corrupt, no deep inheritance chain to trace. Adding a new agent means writing a new file, not modifying existing ones.

---

## The Pipeline: Five Stages, One Shared Context

The Orchestrator runs five agents in sequence. They share a `task` dict that accumulates state as the pipeline progresses:

```
┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌───────────┐
│          │    │          │    │           │    │          │    │           │
│ PLANNER  │───▶│ TRAINER  │───▶│   CODE    │───▶│DEBUGGER  │───▶│ PUBLISHER │
│          │    │          │    │  CHECKER  │    │          │    │           │
│ Discover │    │ Execute  │    │ Validate  │    │ Fix      │    │ HF Hub    │
│ & plan   │    │ experiments   │ config +  │    │ failures │    │ W&B logs  │
│          │    │          │    │ results   │◀───│          │    │           │
└──────────┘    └──────────┘    └───────────┘    └──────────┘    └───────────┘
      │               │                │
      ▼               ▼                ▼
  task["plan"]  task["best_    task["error_
                experiment"]   report"]
```

Each stage reads from and writes to the `task` dict:

| Stage | Reads | Writes |
|---|---|---|
| Planner | available components, CRPS history | `task["plan"]` |
| Trainer | plan | `task["best_experiment"]`, `task["best_metrics"]` |
| CodeChecker | experiment config, run results | validation report |
| Debugger | error report, failed config | fixed `task["experiment"]` |
| Publisher | best experiment, metrics | HF Hub model card, W&B run |

The CodeChecker/Debugger relationship is a loop, not a one-shot step. If the checker flags a problem, the debugger attempts a fix and hands back to the checker. This repeats up to `max_retries` times.

---

## Prompts: Where the Intelligence Lives

If the Python scaffolding is deliberately thin, where does the reasoning quality come from? The prompts.

Prompts in synth-city are not monolithic strings. They're built from **fragments** — named building blocks with priorities, in `pipeline/prompts/`. The `assemble_prompt()` function collects fragments by agent name, sorts by priority, substitutes `{variable}` placeholders from the task context, and concatenates.

```
Fragment library                   Assembled prompt
┌──────────────────────┐           ┌──────────────────────────┐
│ role (priority: 10)  │──────────▶│ [role]                   │
│ phase_1 (priority:20)│──────────▶│ [phase_1 instructions]   │
│ phase_2 (priority:30)│──────────▶│ [phase_2 instructions]   │
│ component_ref (p:40) │──────────▶│ [component reference]    │
│ output_schema (p:50) │──────────▶│ [output format]          │
└──────────────────────┘           └──────────────────────────┘
```

### Phased Reasoning

The Planner uses a two-phase strategy. Phase 1 forces evidence gathering before any planning begins — the agent *must* call `list_blocks()`, `list_heads()`, `compare_results()`, and review CRPS history before proposing anything. Phase 2 produces the concrete plan from that evidence.

This matters because the most common LLM failure mode is hallucinating architectures that don't exist in the framework. Phase 1 eliminates this by making the agent audit reality before reasoning about it.

### Error Catalogs in Prompts

The Debugger's system prompt includes a catalog of known failure patterns with fixes:

```
Config errors:
  - d_model not divisible by num_heads → fix: adjust d_model or num_heads
  - Unknown block name → fix: call list_blocks(), use exact registry key
  - RevIN before positional encoding → fix: swap order

Runtime errors:
  - NaN loss after epoch 1 → fix: reduce learning rate, add gradient clipping
  - CUDA OOM → fix: reduce batch_size or d_model
  - Shape mismatch in head → fix: check head's expected d_model
```

When a new failure mode is discovered, you add a line to the prompt — not a branch to the Python code. The catalog evolves with experience.

---

## Orchestration: Retry, Escalate, Detect Stalls

The Orchestrator wraps every stage (except the Planner) in a retry loop with *escalating temperature*:

```
Attempt 1:  temperature = 0.1   ← focused, deterministic
Attempt 2:  temperature = 0.2   ← slightly more exploratory
Attempt 3:  temperature = 0.3   ← willing to try alternatives
...
```

Low temperature first keeps the agent on track. Higher temperature introduces diversity when it's stuck. This is a heuristic for breaking out of local reasoning attractors — and it works more often than you'd expect.

### Stall Detection

The most pernicious failure mode in the CodeChecker/Debugger loop is *repetitive non-progress*: the Debugger proposes a fix, the fix produces the same broken config, the CodeChecker flags it again, the Debugger proposes the same fix. Infinity.

The Orchestrator detects this by comparing serialised experiment configs across debug attempts:

```python
if current_exp == prev_experiment_json:
    task["user_message"] = (
        "CRITICAL WARNING: The experiment config has NOT changed. "
        "You MUST take a DIFFERENT approach — change blocks, head, "
        "d_model, or learning rate."
    )
```

Simple string comparison, but it catches the most common loop. The injected warning forces the agent to recognise it's stuck and try something genuinely different.

```
Check/Debug loop with stall detection:

  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  ┌───────────┐   FAIL    ┌──────────┐               │
  │  │           │──────────▶│          │               │
  │  │  CODE     │           │ DEBUGGER │               │
  │  │  CHECKER  │◀──────────│          │               │
  │  │           │  fixed    └──────────┘               │
  │  └─────┬─────┘               ▲                      │
  │        │ PASS                │                      │
  │        │           ┌─────────┴────────┐             │
  │        │           │  STALL DETECTED? │             │
  │        │           │  (same config?)  │             │
  │        │           │  → inject warning│             │
  │        │           └──────────────────┘             │
  │        ▼                                            │
  │   → Publisher                        max_retries → FAIL
  └──────────────────────────────────────────────────────┘
```

---

## Two Extension Axes

Here's where the architecture gets interesting. Self-extension operates along two fundamentally different axes.

### Axis 1: Vocabulary Extension

The ComponentAuthor agent can write new backbone blocks or prediction heads directly into the open-synth-miner component directory. After `reload_registry()`, the Trainer's view of available architectures has grown. Nothing else changes.

```
ComponentAuthor writes new block
         │
         ▼
  open-synth-miner/
  components/
    ├── transformer.py
    ├── mamba.py
    └── new_block.py  ◀── written by agent
         │
         ▼
  reload_registry()
         │
         ▼
  Planner now sees new_block in list_blocks()
  Trainer can include it in experiment configs
  All existing heads still work with it (uniform interface)
```

This is **late binding** applied to model architectures. The Trainer's available blocks are not fixed at startup — they're a dynamically resolved set that can grow while the pipeline runs. Any valid block combines freely with any valid head because the interface contract is uniform: `(batch, seq, d_model) → (batch, seq, d_model)`. Full compositionality.

New blocks are **empirically validated** — their quality is determined by CRPS scores in the live competition. The system doesn't need to judge them; the market does.

### Axis 2: Behavioral Extension

The AgentDesigner agent can author entirely new agents — new reasoning strategies with their own toolsets and prompts. A specialized validator for multi-asset architectures, a monitoring agent, a meta-planner.

New agents are **structurally validated** — `ast.parse()` confirms syntax before writing. There's no semantic validation: a written agent could reference tools that don't exist and would fail only at invocation time. The authoring LLM is trusted to produce coherent agents, which is reasonable because the AgentDesigner reads existing agents and prompts before writing new ones.

```
Vocabulary Extension          Behavioral Extension
(model components)            (reasoning strategies)

Compositionality: ✓           Compositionality: ✗
Validation: empirical (CRPS)  Validation: syntactic only
Integration: automatic        Integration: requires Orchestrator edit
Late binding: ✓               Late binding: ✗ (static pipeline)
```

The key asymmetry: you can define ground truth for a model block (does it improve CRPS?) but not for an agent in the abstract (what makes a good Debugger depends on what failures the Trainer produces, which depends on what the Planner proposed). Block quality is context-free; agent quality is relational.

---

## The External Anchor

None of this would be tractable without CRPS scoring against a live prediction market. This is the fixed point around which all self-modification orbits.

Without an external anchor, self-extension becomes an optimization process with no clear objective — prone to reward hacking, metric gaming, and self-reinforcing delusion. The market score is what makes the ComponentAuthor's work meaningful. A new block isn't just syntactically novel; it's empirically evaluated against a competitive environment with real economic stakes.

This suggests a general principle: **the more powerful the self-modification mechanisms, the more important a robust external evaluation signal becomes.** The two scale together.

---

## Strengths

**Architecture search that grows itself.** A conventional hyperparameter sweep explores a predefined grid. synth-city's ComponentAuthor can add entirely new block types mid-research, expanding the search space dynamically based on experimental evidence. The Planner, after seeing what's worked, can request components with specific inductive biases. This is closer to how human researchers actually work.

**Everything is inspectable.** The agent loop is a list of messages. Every tool call is logged. Experiment results are persisted to Hippius (S3-compatible decentralised storage) and tracked in Weights & Biases. When something fails, you can trace exactly what happened and why.

**Adding a tool is a one-liner.** `@tool` decorator, done. No config files, no registration calls, no factory classes. A new contributor can add a capability in 5 minutes without touching any existing code.

**Prompt-driven error handling scales.** Adding a known failure mode to the Debugger's error catalog costs one line in a text file. The equivalent Python `if/else` branch would cost a unit test, a review, and a deploy. The catalog evolves with experience at minimal cost.

**Framework-free simplicity.** A for-loop that any Python developer can read in 15 minutes. No hidden state machines, no DAG compilation, no vendor-specific DSL. When the Orchestrator does something unexpected, you can add a print statement and see exactly what's happening.

---

## Limitations

**The orchestration ceiling.** The `Planner → Trainer → CodeChecker → Debugger → Publisher` chain is imperative code in the Orchestrator. New agents exist outside this chain until a human edits it. Behavioral extension can grow the vocabulary of available agents but cannot change the grammar of how they're assembled into a pipeline. The Orchestrator is a hard constraint that self-modification cannot cross.

**Semantic validation gap for agents.** `ast.parse()` catches syntax errors. It does not catch an agent that calls `run_experiment` when it only has `list_blocks` in its toolset, or one that produces a `finish` payload with the wrong keys. The failure is discovered at runtime, not at write time. This is a weaker guarantee than the empirical validation available for model components.

**Static meta-strategy.** Temperature escalation and stall detection are hand-designed heuristics encoding human intuition about how LLM reasoning fails. They're not learned from experience. If a novel failure mode emerges — one that escalating temperature doesn't resolve — the Orchestrator has no mechanism for noticing the pattern and updating its response. Meta-strategy is a blind spot.

**Tool registration is human-only.** Agents can write model components and other agents. They cannot write and register new tools. This is the one extension pathway that's not self-accessible. A fully self-extending system would include a meta-tool that writes, validates, and registers new tools — a meaningful gap that would require careful sandboxing to implement safely.

**Evaluation latency.** CRPS scoring requires training, validation, and live prediction — potentially hours per experiment. During that latency, the system operates without signal. The ComponentAuthor's decisions can't be rapidly validated. A fast proxy signal would make vocabulary extension dramatically more effective.

---

## What This Pattern Enables

Taking the architecture seriously, what does it actually unlock?

A closed pipeline searches a fixed architecture space. Over N experiments, it finds the best configuration in that space. synth-city's pipeline searches a *growing* space — each ComponentAuthor invocation adds new leaves to the search tree. Over the same N experiments, later ones explore regions that didn't exist when the pipeline started. The search space is itself an output of the research process.

This is, at a small scale, how research groups actually work. The experiments inform what to build next. What gets built expands what can be experimented on. The feedback loop is the point.

Encoding lessons as agents rather than only as prompt edits means the system can accumulate reasoning strategies — not just data. A specialized validation agent written after encountering a class of multi-asset failures captures that institutional knowledge in a reusable, invocable form.

Whether this compound loops into genuine capability improvement depends on how much leverage each extension actually provides. A new block that nudges CRPS by 0.01 on one asset class compounds slowly. A new block that introduces a qualitatively different inductive bias might compound fast. The architecture doesn't guarantee the leverage — it just makes the loop possible.

---

## Summary

synth-city implements what we'd call a **Stratified Extensible Agent System**: three layers (tools, agents, orchestration) with independent extension mechanisms at each layer, late-bound component resolution, and an external evaluation signal as ground truth.

The architecture is most powerful at the vocabulary layer — compositionality is full, validation is empirical, integration is automatic. It's more fragile at the behavioral layer — agents are syntactically validated, integration is manual, and quality guarantees are weaker. The orchestration layer is fixed by design.

The external anchor — CRPS scoring against a live market — is what keeps self-extension coherent. Without it, "improvement" has no meaning. With it, even partial self-extension makes genuine progress: the system can grow its own architecture vocabulary, encode its own reasoning strategies, and let the market judge what it cannot judge about itself.

The ceiling is not insurmountable. Tool registration, orchestration logic, and meta-strategy could all be made extensible through the same patterns already present in the system. Whether they should be is a question about the tradeoff between expressive power and interpretability. The current design resolves that tradeoff conservatively — and perhaps wisely.

---

*synth-city is built on top of [open-synth-miner](https://github.com/tensorlink-dev/open-synth-miner), a composable PyTorch framework for Bittensor Subnet 50. The pipeline uses Chutes AI for LLM inference, Hippius for decentralised storage, and Bittensor Subnet 39 (Basilica) for GPU compute.*
