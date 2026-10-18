from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from typing import Any, TypeVar

from .errors import AgentLoopError, ToolExecutionError
from .tools import Tool, ensure_tool, serialize_result
from .types import AgentEvent, AgentResult, AgentTrace, ChatResponse, StreamEvent, ToolCall

T = TypeVar("T")


class _TraceEvents(list[AgentEvent]):
    def __init__(self, enabled: bool, callback: Any = None) -> None:
        super().__init__()
        self.enabled = enabled
        self.callback = callback

    def append(self, event: AgentEvent) -> None:
        if not self.enabled:
            return
        super().append(event)
        if self.callback is not None:
            self.callback(event)


def _trace_snapshot(events: _TraceEvents, started_at: float, enabled: bool) -> AgentTrace | None:
    if not enabled:
        return None
    return AgentTrace(tuple(events), max(0.0, (time.perf_counter() - started_at) * 1000))


def _parse_structured(text: str, model: type[T] | None) -> T | None:
    if model is None:
        return None
    if hasattr(model, "model_validate_json"):
        return model.model_validate_json(text)  # type: ignore[attr-defined,no-any-return]
    data = json.loads(text)
    if hasattr(model, "parse_obj"):
        return model.parse_obj(data)  # type: ignore[attr-defined,no-any-return]
    return model(**data)  # type: ignore[call-arg,return-value]


def _tool_messages(
    response: ChatResponse,
    tools: dict[str, Tool],
    *,
    error_mode: str,
    trace: list[AgentEvent],
    include_trace: bool,
    iteration: int,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for call in response.tool_calls:
        if include_trace:
            trace.append(AgentEvent("tool_call_started", {
                "iteration": iteration,
                "call_id": call.id,
                "name": call.name,
                "arguments": call.arguments_dict(),
            }))
        started_at = time.perf_counter()
        try:
            spec = tools.get(call.name)
            if spec is None:
                raise ToolExecutionError(f"Unknown tool: {call.name}")
            result = spec.execute(call.arguments)
            success = True
            error = None
        except Exception as exc:
            if error_mode == "raise":
                raise
            result = serialize_result({"ok": False, "error": str(exc)})
            success = False
            error = str(exc)
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.name,
            "content": result,
        })
        if include_trace:
            trace.append(AgentEvent("tool_call_completed", {
                "iteration": iteration,
                "call_id": call.id,
                "name": call.name,
                "result": result,
                "duration_ms": (time.perf_counter() - started_at) * 1000,
                "success": success,
                "error": error,
            }))
    return messages


def _response_from_stream(events: list[StreamEvent]) -> ChatResponse:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    finish_reason: str | None = None
    raw_base: dict[str, Any] = {}
    for event in events:
        raw_base = event.data or raw_base
        if event.text:
            text_parts.append(event.text)
        if event.reasoning:
            reasoning_parts.append(event.reasoning)
        choices = event.data.get("choices") or []
        if choices and isinstance(choices[0], Mapping):
            finish_reason = choices[0].get("finish_reason") or finish_reason
        for call in event.tool_calls:
            state = calls.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
            if call.id:
                state["id"] = call.id
            if call.name:
                state["name"] += call.name
            if call.arguments:
                state["arguments"] += call.arguments
    tool_calls = [
        {"id": value["id"], "type": "function", "function": {"name": value["name"], "arguments": value["arguments"]}}
        for _, value in sorted(calls.items())
    ]
    message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return ChatResponse({
        "id": raw_base.get("id"),
        "model": raw_base.get("model"),
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": raw_base.get("usage") or {},
    })


class Agent:
    """A small tool-execution loop built on persisted runtime conversations.

    The runtime stores assistant tool-call messages and tool-result messages.
    The SDK therefore sends only the new user or tool messages for each round.
    """

    def __init__(
        self,
        client: Any,
        *,
        instructions: str | None = None,
        system_prompt: str | None = None,
        tools: Iterable[Any] = (),
        max_iterations: int = 8,
        tool_choice: str = "auto",
        max_output_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        reasoning_effort: str | None = None,
        tool_error_mode: str = "message",
        trace_callback: Any = None,
        title: str | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if tool_error_mode not in {"message", "raise"}:
            raise ValueError("tool_error_mode must be 'message' or 'raise'")
        self.client = client
        self.instructions = instructions if instructions is not None else system_prompt
        self.max_iterations = max_iterations
        self.tool_choice = tool_choice
        self.defaults = {
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "reasoning_effort": reasoning_effort,
        }
        self.tool_error_mode = tool_error_mode
        self.trace_callback = trace_callback
        self.title = title
        self.tools: dict[str, Tool] = {}
        self._conversation = None
        self.last_result: AgentResult | None = None
        self.add_tools(tools)

    def add_tools(self, tools: Iterable[Any]) -> None:
        for value in tools:
            spec = ensure_tool(value)
            self.tools[spec.name] = spec

    def _ensure_conversation(self):
        if self._conversation is None:
            self._conversation = self.client.conversations.create(
                title=self.title,
                system=self.instructions,
            )
        return self._conversation

    def run(
        self,
        prompt: str,
        *,
        trace: bool = False,
        response_model: type[T] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        started_at = time.perf_counter()
        conversation = self._ensure_conversation()
        params = {key: value for key, value in self.defaults.items() if value is not None}
        params.update(kwargs)
        if response_model is not None:
            params.setdefault("response_format", {"type": "json_object"})
        tool_definitions = [spec.definition() for spec in self.tools.values()]
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        trace_events = _TraceEvents(trace, self.trace_callback)
        if trace:
            trace_events.append(AgentEvent("agent_started", {"prompt": prompt}))

        for iteration in range(1, self.max_iterations + 1):
            request = dict(params)
            if tool_definitions:
                request["tools"] = tool_definitions
                request["tool_choice"] = self.tool_choice
            response = conversation.complete(messages, **request)
            if trace:
                trace_events.append(AgentEvent("model_response", {
                    "iteration": iteration,
                    "finish_reason": response.finish_reason,
                    "text": response.text,
                }))
            if not response.tool_calls:
                parsed = _parse_structured(response.text, response_model)
                if trace:
                    trace_events.append(AgentEvent("agent_completed", {"iteration": iteration}))
                result = AgentResult(response.text, response, conversation.id, _trace_snapshot(trace_events, started_at, trace), parsed)
                self.last_result = result
                return result
            messages = _tool_messages(
                response,
                self.tools,
                error_mode=self.tool_error_mode,
                trace=trace_events,
                include_trace=trace,
                iteration=iteration,
            )
        raise AgentLoopError(f"Agent exceeded max_iterations={self.max_iterations}")

    def stream(self, prompt: str, *, trace: bool = False, **kwargs: Any) -> Iterator[StreamEvent | AgentEvent]:
        started_at = time.perf_counter()
        conversation = self._ensure_conversation()
        params = {key: value for key, value in self.defaults.items() if value is not None}
        params.update(kwargs)
        tool_definitions = [spec.definition() for spec in self.tools.values()]
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        trace_events = _TraceEvents(trace, self.trace_callback)
        if trace:
            trace_events.append(AgentEvent("agent_started", {"prompt": prompt}))
        for iteration in range(1, self.max_iterations + 1):
            request = dict(params)
            request["stream"] = True
            if tool_definitions:
                request["tools"] = tool_definitions
                request["tool_choice"] = self.tool_choice
            received: list[StreamEvent] = []
            for event in conversation.stream_messages(messages, **request):
                received.append(event)
                yield event
            response = _response_from_stream(received)
            if not response.tool_calls:
                if trace:
                    trace_events.append(AgentEvent("agent_completed", {"iteration": iteration}))
                self.last_result = AgentResult(response.text, response, conversation.id, _trace_snapshot(trace_events, started_at, trace))
                return
            messages = []
            for call in response.tool_calls:
                started = AgentEvent("tool_call_started", {
                    "iteration": iteration, "call_id": call.id, "name": call.name,
                    "arguments": call.arguments_dict(),
                })
                if trace:
                    trace_events.append(started)
                    yield started
                started_at = time.perf_counter()
                try:
                    spec = self.tools.get(call.name)
                    if spec is None:
                        raise ToolExecutionError(f"Unknown tool: {call.name}")
                    content = spec.execute(call.arguments)
                    success, error = True, None
                except Exception as exc:
                    if self.tool_error_mode == "raise":
                        raise
                    content = serialize_result({"ok": False, "error": str(exc)})
                    success, error = False, str(exc)
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content})
                completed = AgentEvent("tool_call_completed", {
                    "iteration": iteration, "call_id": call.id, "name": call.name,
                    "result": content, "duration_ms": (time.perf_counter() - started_at) * 1000,
                    "success": success, "error": error,
                })
                if trace:
                    trace_events.append(completed)
                    yield completed
        raise AgentLoopError(f"Agent exceeded max_iterations={self.max_iterations}")

    def reset(self) -> None:
        # Keep the previous server-side transcript for inspection; a new run will
        # receive a new persisted conversation.
        self._conversation = None
        self.last_result = None


class AsyncAgent:
    """Asynchronous counterpart to :class:`Agent`, including async tools."""

    def __init__(
        self,
        client: Any,
        *,
        instructions: str | None = None,
        system_prompt: str | None = None,
        tools: Iterable[Any] = (),
        max_iterations: int = 8,
        tool_choice: str = "auto",
        max_output_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        reasoning_effort: str | None = None,
        tool_error_mode: str = "message",
        trace_callback: Any = None,
        title: str | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if tool_error_mode not in {"message", "raise"}:
            raise ValueError("tool_error_mode must be 'message' or 'raise'")
        self.client = client
        self.instructions = instructions if instructions is not None else system_prompt
        self.max_iterations = max_iterations
        self.tool_choice = tool_choice
        self.defaults = {
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "reasoning_effort": reasoning_effort,
        }
        self.tool_error_mode = tool_error_mode
        self.trace_callback = trace_callback
        self.title = title
        self.tools: dict[str, Tool] = {}
        self._conversation = None
        self.last_result: AgentResult | None = None
        self.add_tools(tools)

    def add_tools(self, tools: Iterable[Any]) -> None:
        for value in tools:
            spec = ensure_tool(value)
            self.tools[spec.name] = spec

    async def _ensure_conversation(self):
        if self._conversation is None:
            self._conversation = await self.client.conversations.create(
                title=self.title,
                system=self.instructions,
            )
        return self._conversation

    async def run(
        self,
        prompt: str,
        *,
        trace: bool = False,
        response_model: type[T] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        started_at = time.perf_counter()
        conversation = await self._ensure_conversation()
        params = {key: value for key, value in self.defaults.items() if value is not None}
        params.update(kwargs)
        if response_model is not None:
            params.setdefault("response_format", {"type": "json_object"})
        tool_definitions = [spec.definition() for spec in self.tools.values()]
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        trace_events = _TraceEvents(trace, self.trace_callback)
        if trace:
            trace_events.append(AgentEvent("agent_started", {"prompt": prompt}))
        for iteration in range(1, self.max_iterations + 1):
            request = dict(params)
            if tool_definitions:
                request["tools"] = tool_definitions
                request["tool_choice"] = self.tool_choice
            response = await conversation.complete(messages, **request)
            if trace:
                trace_events.append(AgentEvent("model_response", {
                    "iteration": iteration, "finish_reason": response.finish_reason, "text": response.text,
                }))
            if not response.tool_calls:
                parsed = _parse_structured(response.text, response_model)
                if trace:
                    trace_events.append(AgentEvent("agent_completed", {"iteration": iteration}))
                result = AgentResult(response.text, response, conversation.id, _trace_snapshot(trace_events, started_at, trace), parsed)
                self.last_result = result
                return result
            messages = []
            for call in response.tool_calls:
                if trace:
                    trace_events.append(AgentEvent("tool_call_started", {
                        "iteration": iteration, "call_id": call.id, "name": call.name,
                        "arguments": call.arguments_dict(),
                    }))
                started_at = time.perf_counter()
                try:
                    spec = self.tools.get(call.name)
                    if spec is None:
                        raise ToolExecutionError(f"Unknown tool: {call.name}")
                    content = await spec.aexecute(call.arguments)
                    success, error = True, None
                except Exception as exc:
                    if self.tool_error_mode == "raise":
                        raise
                    content = serialize_result({"ok": False, "error": str(exc)})
                    success, error = False, str(exc)
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content})
                if trace:
                    trace_events.append(AgentEvent("tool_call_completed", {
                        "iteration": iteration, "call_id": call.id, "name": call.name,
                        "result": content, "duration_ms": (time.perf_counter() - started_at) * 1000,
                        "success": success, "error": error,
                    }))
        raise AgentLoopError(f"Agent exceeded max_iterations={self.max_iterations}")

    async def stream(self, prompt: str, *, trace: bool = False, **kwargs: Any) -> AsyncIterator[StreamEvent | AgentEvent]:
        started_at = time.perf_counter()
        conversation = await self._ensure_conversation()
        params = {key: value for key, value in self.defaults.items() if value is not None}
        params.update(kwargs)
        tool_definitions = [spec.definition() for spec in self.tools.values()]
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        trace_events = _TraceEvents(trace, self.trace_callback)
        if trace:
            trace_events.append(AgentEvent("agent_started", {"prompt": prompt}))
        for iteration in range(1, self.max_iterations + 1):
            request = dict(params)
            request["stream"] = True
            if tool_definitions:
                request["tools"] = tool_definitions
                request["tool_choice"] = self.tool_choice
            received: list[StreamEvent] = []
            async for event in conversation.stream_messages(messages, **request):
                received.append(event)
                yield event
            response = _response_from_stream(received)
            if not response.tool_calls:
                if trace:
                    trace_events.append(AgentEvent("agent_completed", {"iteration": iteration}))
                self.last_result = AgentResult(response.text, response, conversation.id, _trace_snapshot(trace_events, started_at, trace))
                return
            messages = []
            for call in response.tool_calls:
                started = AgentEvent("tool_call_started", {
                    "iteration": iteration, "call_id": call.id, "name": call.name,
                    "arguments": call.arguments_dict(),
                })
                if trace:
                    trace_events.append(started)
                    yield started
                started_at = time.perf_counter()
                try:
                    spec = self.tools.get(call.name)
                    if spec is None:
                        raise ToolExecutionError(f"Unknown tool: {call.name}")
                    content = await spec.aexecute(call.arguments)
                    success, error = True, None
                except Exception as exc:
                    if self.tool_error_mode == "raise":
                        raise
                    content = serialize_result({"ok": False, "error": str(exc)})
                    success, error = False, str(exc)
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content})
                completed = AgentEvent("tool_call_completed", {
                    "iteration": iteration, "call_id": call.id, "name": call.name,
                    "result": content, "duration_ms": (time.perf_counter() - started_at) * 1000,
                    "success": success, "error": error,
                })
                if trace:
                    trace_events.append(completed)
                    yield completed
        raise AgentLoopError(f"Agent exceeded max_iterations={self.max_iterations}")

    def reset(self) -> None:
        self._conversation = None
        self.last_result = None
