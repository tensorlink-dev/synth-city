"""
SimpleAgent — the ~200-line backbone of the agentic pipeline.

Implements a dead-simple for-loop:
    1. Send messages to the LLM.
    2. If the response contains tool calls, execute them.
    3. Append tool results to the conversation.
    4. Repeat until the agent calls the special ``finish`` tool or we hit
       *max_turns*.

All sophistication lives in the **prompts** and **orchestration**, not here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import BaseModel

from pipeline.monitor import get_monitor
from pipeline.providers.chutes_client import chat_completion_with_backoff

logger = logging.getLogger(__name__)
_mon = get_monitor()

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
ToolFunction = Callable[..., Any]


@dataclass
class ToolResult:
    """Wraps the return value of a tool invocation."""

    tool_call_id: str
    name: str
    content: str
    is_finish: bool = False
    structured: Any = None


@dataclass
class AgentResult:
    """Final output of an agent run."""

    success: bool
    structured: Any = None
    raw_text: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns_used: int = 0


# ---------------------------------------------------------------------------
# Argument coercion helpers
# ---------------------------------------------------------------------------

def _coerce_args(params: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Robustly coerce sloppy LLM outputs to the expected types.

    Handles: empty-string → empty-list, JSON-in-string, string booleans.
    """
    properties = schema.get("properties", {})
    coerced: dict[str, Any] = {}
    for key, value in params.items():
        expected = properties.get(key, {})
        expected_type = expected.get("type")

        if expected_type == "array" and isinstance(value, str):
            value = value.strip()
            if value == "":
                value = []
            else:
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    value = [value]

        elif expected_type == "boolean" and isinstance(value, str):
            value = value.lower() in ("true", "1", "yes")

        elif expected_type == "integer" and isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                pass

        elif expected_type == "number" and isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass

        elif isinstance(value, str):
            # Try to parse JSON objects/arrays hiding in strings
            stripped = value.strip()
            if stripped and stripped[0] in ("{", "["):
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError:
                    pass

        coerced[key] = value
    return coerced


# ---------------------------------------------------------------------------
# SimpleAgent
# ---------------------------------------------------------------------------

class SimpleAgent:
    """Minimal multi-turn tool-use agent.

    Parameters
    ----------
    client : OpenAI
        An OpenAI-compatible client (works with Chutes, OpenRouter, etc.).
    model : str
        Model identifier.
    system_prompt : str
        The system-level instruction.
    tools : dict[str, ToolFunction]
        name → callable mapping.  Each callable's docstring + annotations
        are used to build the tool schema (or pass *tool_schemas* explicitly).
    tool_schemas : list[dict] | None
        Explicit OpenAI-format tool schemas.  If ``None`` the agent will try
        to auto-generate them from *tools*.
    max_turns : int
        Safety cap on conversation turns.
    temperature : float
        Sampling temperature for the LLM.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        system_prompt: str,
        tools: dict[str, ToolFunction],
        tool_schemas: list[dict[str, Any]] | None = None,
        max_turns: int = 50,
        temperature: float = 0.1,
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.tools = dict(tools)
        self.tool_schemas = tool_schemas or []
        self.max_turns = max_turns
        self.temperature = temperature

        # Inject the built-in finish tool
        self._finish_schema = {
            "type": "function",
            "function": {
                "name": "finish",
                "description": (
                    "Signal that the task is complete.  Pass a JSON object with "
                    "'success' (bool) and optionally 'result' (any structured data) "
                    "and 'summary' (text)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "success": {
                            "type": "boolean",
                            "description": "Whether the task succeeded.",
                        },
                        "result": {
                            "type": "string",
                            "description": "Structured result as JSON string.",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Human-readable summary.",
                        },
                    },
                    "required": ["success"],
                },
            },
        }

    # ------------------------------------------------------------------ run
    def run(self, user_message: str, context: list[dict[str, Any]] | None = None) -> AgentResult:
        """Execute the agent loop.

        Parameters
        ----------
        user_message : str
            The initial user turn.
        context : list[dict] | None
            Optional prior messages to prepend (e.g. from a previous agent).
        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": user_message})

        all_schemas = self.tool_schemas + [self._finish_schema]
        tool_names = sorted(self.tools.keys())
        logger.info(
            "Agent starting: model=%s  tools=%d [%s]  max_turns=%d",
            self.model, len(tool_names), ", ".join(tool_names), self.max_turns,
        )
        _mon.emit(
            "agent", "agent_start",
            name=self.model, model=self.model,
            tools=tool_names, max_turns=self.max_turns,
        )

        for turn in range(self.max_turns):
            logger.debug("turn %d  model=%s  msgs=%d", turn, self.model, len(messages))
            _mon.emit("agent", "agent_turn", turn=turn + 1)

            try:
                response = chat_completion_with_backoff(
                    self.client,
                    model=self.model,
                    messages=messages,
                    tools=all_schemas if all_schemas else None,
                    temperature=self.temperature,
                )
            except Exception as exc:
                logger.error("LLM call failed on turn %d after retries: %s", turn, exc)
                return AgentResult(
                    success=False,
                    raw_text=f"LLM call failed: {type(exc).__name__}: {exc}",
                    messages=messages,
                    turns_used=turn + 1,
                )
            choice = response.choices[0]
            assistant_msg: dict[str, Any] = {"role": "assistant"}

            if choice.message.content:
                assistant_msg["content"] = choice.message.content

            # ---- no tool calls → done (implicit finish)
            if not choice.message.tool_calls:
                assistant_msg.setdefault("content", "")
                messages.append(assistant_msg)
                _mon.emit("agent", "agent_finish", success=True, turns=turn + 1)
                return AgentResult(
                    success=True,
                    raw_text=choice.message.content or "",
                    messages=messages,
                    turns_used=turn + 1,
                )

            # ---- execute tool calls
            assistant_msg["tool_calls"] = [
                tc.model_dump() for tc in choice.message.tool_calls
            ]
            messages.append(assistant_msg)

            for tc in choice.message.tool_calls:
                result = self._execute_tool_call(tc)
                messages.append({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                })
                if result.is_finish:
                    finish_success = (
                        result.structured.get("success", True)
                        if isinstance(result.structured, dict)
                        else True
                    )
                    _mon.emit("agent", "agent_finish", success=finish_success, turns=turn + 1)
                    return AgentResult(
                        success=finish_success,
                        structured=result.structured,
                        raw_text=result.content,
                        messages=messages,
                        turns_used=turn + 1,
                    )

        # Exhausted turns
        logger.warning("Agent exhausted %d turns without finishing", self.max_turns)
        _mon.emit("agent", "agent_finish", success=False, turns=self.max_turns)
        return AgentResult(
            success=False,
            raw_text="max turns exhausted",
            messages=messages,
            turns_used=self.max_turns,
        )

    # --------------------------------------------------------- tool dispatch
    def _execute_tool_call(self, tc: ChatCompletionMessageToolCall) -> ToolResult:
        name = tc.function.name
        try:
            raw_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            raw_args = {}

        # Handle the built-in finish tool
        if name == "finish":
            structured = raw_args
            # Try to parse the "result" field if it's JSON
            if "result" in structured and isinstance(structured["result"], str):
                try:
                    structured["result"] = json.loads(structured["result"])
                except json.JSONDecodeError:
                    pass
            logger.info(
                "Agent finishing: success=%s summary=%s",
                structured.get("success"),
                str(structured.get("summary", ""))[:200],
            )
            return ToolResult(
                tool_call_id=tc.id,
                name="finish",
                content=json.dumps(structured),
                is_finish=True,
                structured=structured,
            )

        func = self.tools.get(name)
        if func is None:
            available = ", ".join(sorted(self.tools))
            logger.warning("Tool call for unknown tool: %s (available: %s)", name, available)
            return ToolResult(
                tool_call_id=tc.id,
                name=name,
                content=json.dumps({"error": f"Unknown tool: {name}"}),
            )

        # Coerce arguments if the tool has a schema
        schema = self._schema_for(name)
        if schema:
            raw_args = _coerce_args(raw_args, schema)

        logger.info("Tool call: %s(%s)", name, json.dumps(raw_args, default=str)[:200])
        _mon.emit("tool", "tool_call", name=name)
        try:
            result = func(**raw_args)
            # If the tool returns a Pydantic model, serialize it
            if isinstance(result, BaseModel):
                content = result.model_dump_json()
            elif isinstance(result, (dict, list)):
                content = json.dumps(result)
            else:
                content = str(result)
            logger.info("Tool %s returned %d chars", name, len(content))
            _mon.emit("tool", "tool_result", name=name, size=len(content))
        except Exception as exc:
            logger.exception("Tool %s raised an exception", name)
            content = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
            _mon.emit("tool", "tool_result", name=name, size=len(content), error=True)

        return ToolResult(tool_call_id=tc.id, name=name, content=content)

    def _schema_for(self, name: str) -> dict[str, Any]:
        for s in self.tool_schemas:
            if s.get("function", {}).get("name") == name:
                return s["function"].get("parameters", {})
        return {}

    # ------------------------------------------------------ context helpers
    def inject_message(self, messages: list[dict], role: str, content: str) -> None:
        """Inject a message into an existing conversation (e.g. stall warnings)."""
        messages.append({"role": role, "content": content})
