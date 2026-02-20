# synth-city: Philosophy & Architectural Approach

This document walks through the design philosophy behind synth-city — **why** the
codebase is structured the way it is, what trade-offs were made, and how the
pieces fit together.

---

## The Core Problem

Bittensor Subnet 50 (Synth) rewards miners who produce well-calibrated
probabilistic price forecasts. The competition metric is **CRPS** (Continuous
Ranked Probability Score) — a measure of how closely a predicted probability
distribution matches reality. Winning requires generating 1,000 Monte Carlo
price paths per asset at 5-minute intervals over 24 hours, across multiple
assets (BTC, ETH, SOL, gold, equities).

Manually discovering good model architectures, training them, debugging
failures, and publishing winners is slow and error-prone. synth-city automates
that entire loop by chaining specialized AI agents together.

---

## Design Philosophy: Simplicity as a Feature

The single most important design principle in synth-city is:

> **All sophistication lives in the prompts and orchestration, not in framework code.**

This is stated directly in the SimpleAgent source and it drives every major
architectural decision. The Python code is deliberately minimal — a thin
scaffolding that lets LLM agents do the heavy lifting through well-crafted
prompts. When you need the system to behave differently, you change a prompt
fragment, not a class hierarchy.

### Why Not a Framework?

There are many agent frameworks available — LangGraph, AutoGPT, CrewAI, and
others. synth-city deliberately avoids them. The reasons:

- **Readability over abstraction.** The core agent loop is a ~200-line
  `for`-loop in `simple_agent.py`. Anyone who knows Python and the OpenAI chat
  API can read it, step through it, and understand exactly what happens on every
  turn. Frameworks introduce DSLs, task graphs, and planning layers that obscure
  the actual execution flow.

- **Debuggability.** When an agent misbehaves, you print the message history.
  There are no hidden state machines, no DAG compilation steps, no task queues
  to inspect. The entire agent state is a list of messages.

- **No vendor lock-in.** The LLM client (`chutes_client.py`) speaks the
  OpenAI-compatible API. Swapping providers means changing one URL and one API
  key. No framework-specific adapters needed.

- **Stability.** Agent frameworks evolve rapidly and often introduce breaking
  changes. A for-loop doesn't break between releases.

The philosophy is: if you can solve a problem with 200 lines of straightforward
Python, don't solve it with 2,000 lines of framework.

---

## The Agent Loop: A For-Loop, Not a DAG

The heart of synth-city is `SimpleAgent.run()` in
`pipeline/providers/simple_agent.py`. The loop is:

```
for turn in range(max_turns):
    response = llm.chat(messages, tools)
    if no tool calls in response:
        return response (agent is done thinking)
    for each tool call:
        result = execute_tool(tool_call)
        append result to messages
        if result signals "finish":
            return structured result
return timeout
```

This mirrors how LLM conversations naturally work — turn-taking with iterative
refinement. The agent thinks, acts (via tool calls), observes (via tool
results), and repeats. There's no need for a planning graph because the LLM
**is** the planner. The prompts guide its reasoning; the loop just keeps the
conversation going.

Two termination modes exist:

1. **Implicit finish**: The LLM responds with no tool calls. It has nothing
   more to do.
2. **Explicit finish**: The LLM calls the built-in `finish` tool with a
   structured result payload.

This dual approach handles both chatty models (that narrate their conclusion)
and disciplined models (that signal completion programmatically).

---

## Tool Registry: One Decorator, No Config Files

Adding a capability to an agent should be trivial. In synth-city, it is:

```python
@tool(description="Fetch the latest price data for an asset")
def fetch_prices(asset: str, interval: str = "5m") -> str:
    ...
```

That's it. The `@tool` decorator in `pipeline/tools/registry.py`:

1. Registers the function in a global dictionary.
2. Infers the JSON schema from Python type hints (`str` → `"string"`,
   `int` → `"integer"`, `list` → `"array"`, etc.).
3. Marks parameters without defaults as required.

No configuration files. No factory classes. No registration calls. The tool
exists the moment the module is imported.

### Dynamic Scoping

Not every agent should see every tool. The Planner needs discovery tools
(list blocks, review history). The Trainer needs experiment tools (create,
run, validate). The Debugger needs both plus error-analysis tools.

Each agent's `build_tools()` method returns a list of tool names:

```python
def build_tools(self, task):
    return build_toolset("list_blocks", "list_heads", "compare_results", ...)
```

`build_toolset()` pulls only those tools from the global registry. This keeps
each agent's action space focused — a Planner that can't accidentally start
training, a Publisher that can't accidentally create experiments.

### Argument Coercion: Tolerating LLM Sloppiness

LLMs produce imperfect JSON. A parameter typed as `list` might arrive as
`"[]"` (a string), `[[...]]` (double-nested), or `""` (empty string). A boolean
might arrive as `"true"` instead of `true`.

Rather than making every tool handle these edge cases, the `_coerce_args()`
function in `simple_agent.py` normalises arguments **before** dispatching to the
tool:

- Empty string → empty list (when list expected)
- JSON-in-string → parsed JSON
- String booleans → real booleans
- String numbers → real numbers

This centralised coercion means tool authors write clean Python functions with
real types. The messiness of LLM output is handled once, in one place.

---

## Agent Composition: Thin Wrappers, Not Deep Hierarchies

Every agent in synth-city subclasses `BaseAgentWrapper`
(`pipeline/agents/base.py`), which defines exactly four hooks:

| Hook | Purpose |
|---|---|
| `build_system_prompt(task)` | Assemble the agent's personality and instructions |
| `build_tools(task)` | Select which tools this agent can use |
| `build_context(task)` | Inject prior conversation or data as context messages |
| `post_process(result, task)` | Clean up or transform the agent's output |

That's the entire interface. An agent implementation looks like this:

```python
class PlannerAgent(BaseAgentWrapper):
    agent_name = "planner"

    def build_system_prompt(self, task):
        return assemble_prompt("planner", task.get("channel"), task)

    def build_tools(self, task):
        return build_toolset("list_blocks", "list_heads", "list_presets", ...)

    def build_context(self, task):
        context = []
        if "crps_scores" in task:
            context.append({"role": "user", "content": format_scores(task)})
        return context
```

### Why Composition Over Inheritance?

- **No surprise overrides.** Each agent independently declares what it needs.
  The Trainer's tools have nothing to do with the Debugger's tools. There's no
  shared `super().build_tools()` to worry about.

- **Easy to test.** Provide a `task` dict, call `agent.run()`, inspect the
  result. No complex setup, no mocking base-class behaviour.

- **Parallel development.** New agents can be added without touching existing
  ones. Write a new file in `pipeline/agents/`, add a prompt module in
  `pipeline/prompts/`, and you're done.

- **Flat is better than nested.** Python's Zen applies here. A one-level
  hierarchy is easy to reason about. Deep inheritance chains hide behaviour in
  parent classes and make it hard to predict what a method call actually does.

---

## Prompt Engineering: Where the Intelligence Lives

If the Python code is deliberately simple, where does the intelligence come
from? The prompts.

### Composable Fragments

Prompts in synth-city are not monolithic strings. They're built from
**fragments** — named building blocks with priorities, stored in
`pipeline/prompts/`. The `assemble_prompt()` function collects fragments by
agent name and channel, sorts by priority, and concatenates them.

This means:

- **Fragments are reusable.** A "role" fragment can be shared across channels
  (default, experimental).
- **Fragments are swappable.** Change the Planner's Phase 1 instructions without
  touching Phase 2.
- **Fragments support variable substitution.** `{variable}` placeholders get
  filled from the task context at assembly time.

### Phased Reasoning

The Planner agent uses a two-phase prompting strategy:

**Phase 1 — Diagnostic:** Before proposing anything, the agent must call
discovery tools: list available backbone blocks, review presets, check prior
experiment results, analyse historical data, and identify gaps. This prevents
the LLM from hallucinating architectures that don't exist in the framework.

**Phase 2 — Execution Plan:** Only after gathering evidence does the agent
produce a concrete plan with architecture decisions, hyperparameter ranges,
experiment priorities, and success criteria.

This structure forces the LLM to think before acting — a pattern that
dramatically improves output quality compared to asking for a plan in a single
shot.

### Error Catalogs

The Debugger agent's prompt includes a catalog of known error patterns with
solutions:

- **Config errors:** `d_model` not divisible by attention heads, unknown block
  names, incorrect RevIN placement.
- **Execution errors:** Out-of-memory, NaN loss, shape mismatches, infinite
  CRPS.
- **Performance issues:** CRPS worse than baseline.

Each pattern includes a diagnosis and fix. This is **prompt-driven error
handling** — rather than writing Python `if/else` chains for every failure mode,
the prompt teaches the agent to recognise patterns and apply fixes. When a new
failure mode appears, you add a line to the prompt, not a branch to the code.

### Validation Checklists

The CodeChecker agent uses a structured checklist covering config validity,
architecture sanity, composition rules, training configuration, and results
validation. The agent works through each category methodically, producing a
pass/fail verdict with specific reasons.

This is more robust than programmatic validation for the same reason: the
checklist evolves with experience, and the LLM can catch subtle issues
(questionable hyperparameter combinations, architectures unlikely to converge)
that hard-coded rules would miss.

---

## Orchestration: Retry, Escalate, Detect Stalls

The `PipelineOrchestrator` in `pipeline/orchestrator.py` chains agents together
with built-in resilience:

### The Pipeline Flow

```
Planner → Trainer → CodeChecker → (Debugger ↔ CodeChecker) → Publisher
```

Each stage receives the `task` dict — a shared context that accumulates data as
the pipeline progresses:

| Stage | Reads | Writes |
|---|---|---|
| Planner | available components, history | `task["plan"]` |
| Trainer | plan | `task["best_experiment"]`, `task["best_metrics"]` |
| CodeChecker | experiment config, run results | validation report |
| Debugger | error report, failed config | fixed `task["experiment"]` |
| Publisher | best experiment, metrics | HF Hub model, W&B logs |

### Temperature Escalation

When an agent fails, the orchestrator retries with increasing temperature:

```
Attempt 1: temperature = 0.1  (focused, deterministic)
Attempt 2: temperature = 0.2  (slightly more creative)
Attempt 3: temperature = 0.3  (exploring alternatives)
...
```

Low temperature first keeps the agent on track. If it gets stuck, higher
temperature introduces diversity — the agent tries different approaches rather
than repeating the same failing strategy.

### Stall Detection

The Debugger can fall into a loop: propose a fix, fail validation, propose the
same fix again. The orchestrator detects this by comparing experiment configs
between debug attempts. If the config hasn't changed, it injects a critical
warning:

> "CRITICAL WARNING: You MUST take a DIFFERENT approach. Your previous fix
> produced the same configuration."

This forces the agent out of its rut. It's a simple mechanism — just string
comparison of serialised configs — but it prevents the most common failure mode
in agentic loops: repetitive, unproductive retries.

---

## Configuration: Environment Variables, Not Config Files

All settings flow through `.env`, accessed via `config.py`:

- **API keys and endpoints** — LLM provider, HuggingFace Hub, Weights & Biases,
  Hippius storage.
- **Per-agent model selection** — `PLANNER_MODEL`, `TRAINER_MODEL`,
  `DEBUGGER_MODEL`, etc. The Planner might use a large reasoning model while the
  CodeChecker uses a smaller, faster one.
- **Pipeline parameters** — max retries, temperature ranges, training epochs.

### Why Environment Variables?

- **No code changes between environments.** Local development, cloud GPU,
  production — same code, different `.env`.
- **No secrets in source control.** `.env` is gitignored. API keys never touch
  the repository.
- **Simple override.** `PLANNER_MODEL=gpt-4 python main.py pipeline` — one
  environment variable changes the model for a single run.

---

## The Research Session Abstraction

Agents never interact directly with PyTorch or the open-synth-miner internals.
Instead, they use the **ResearchSession** API exposed through tools:

- `create_experiment(blocks, head, d_model, ...)` — build a config
- `validate_experiment(config)` — check for errors before training
- `run_experiment(config, epochs)` — train and evaluate
- `compare_results()` — rank all experiments by CRPS

This abstraction serves three purposes:

1. **Safety.** Agents can't corrupt internal state or produce invalid PyTorch
   code. The API validates inputs and manages the training lifecycle.

2. **Swappability.** Training can happen locally or on remote GPUs (via the
   Basilica compute client for Bittensor Subnet 39). The agent doesn't know or
   care where computation happens.

3. **Reproducibility.** Same config, same results. The session manages seeds,
   data loading, and evaluation consistently.

---

## What This Architecture Optimises For

synth-city's architecture is not trying to be a general-purpose agent framework.
It's optimised for a specific set of properties:

**Clarity.** A new contributor can read `simple_agent.py` in 15 minutes and
understand the entire agent execution model. There are no hidden state machines
or implicit behaviours.

**Modularity.** New agents, tools, and prompts can be added independently. The
tool registry, agent composition pattern, and fragment-based prompts all support
this.

**Iterability.** The fastest way to improve agent behaviour is to edit prompts.
Error catalogs, validation checklists, and phased reasoning all live in text
files that can be updated without touching Python.

**Resilience.** Temperature escalation, stall detection, and the
CodeChecker/Debugger loop handle failures gracefully. The system doesn't crash
on the first error — it adapts.

**Transparency.** Every agent's input and output is inspectable JSON. The
orchestrator logs every stage. Experiment results are persisted to decentralised
storage (Hippius) and tracked in Weights & Biases. Nothing happens in a black
box.

---

## Summary

The philosophy can be distilled to five principles:

1. **Keep the framework thin.** A for-loop beats a DAG. A decorator beats a
   config file. A flat hierarchy beats a deep one.

2. **Put intelligence in prompts.** Phased reasoning, error catalogs, and
   validation checklists are more flexible and more maintainable than equivalent
   Python logic.

3. **Scope tightly.** Each agent sees only the tools it needs. Each tool does
   one thing. Each prompt fragment serves one purpose.

4. **Handle failure gracefully.** Retry with escalation. Detect stalls. Give
   the agent another chance with more creativity before giving up.

5. **Make everything inspectable.** JSON in, JSON out. Logged stages. Persisted
   results. When something goes wrong, you can trace exactly what happened and
   why.

These principles emerged from practical experience building autonomous AI
systems: simplicity survives contact with reality; complexity doesn't.
