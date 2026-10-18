from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


JSON = dict[str, Any]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}



@dataclass(slots=True)
class ChatMessage:
    role: str
    content: Any | None = None
    name: str | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[JSON] | None = None

    def to_dict(self) -> JSON:
        result: JSON = {"role": self.role}
        for key in ("content", "name", "tool_call_id", "reasoning_content", "tool_calls"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def system(cls, content: Any) -> "ChatMessage":
        return cls("system", content)

    @classmethod
    def user(cls, content: Any) -> "ChatMessage":
        return cls("user", content)

    @classmethod
    def assistant(cls, content: Any | None = None, **kwargs: Any) -> "ChatMessage":
        return cls("assistant", content, **kwargs)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, *, name: str | None = None) -> "ChatMessage":
        return cls("tool", content, name=name, tool_call_id=tool_call_id)


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str = "{}"
    index: int = 0

    @classmethod
    def from_response(cls, value: Mapping[str, Any], *, index: int = 0) -> "ToolCall":
        function = _mapping(value.get("function"))
        return cls(
            id=str(value.get("id") or ""),
            name=str(function.get("name") or ""),
            arguments=str(function.get("arguments") or "{}"),
            index=int(value.get("index", index)),
        )

    def arguments_dict(self) -> JSON:
        import json

        try:
            value = json.loads(self.arguments)
        except (json.JSONDecodeError, TypeError):
            return {}
        return value if isinstance(value, dict) else {}


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Typed convenience view over a runtime/OpenAI-compatible response."""

    raw: JSON

    @property
    def choices(self) -> list[JSON]:
        return list(self.raw.get("choices") or [])

    @property
    def message(self) -> JSON:
        choices = self.choices
        if not choices:
            raise ValueError("Chat response contains no choices")
        return dict(_mapping(choices[0].get("message")))

    @property
    def text(self) -> str:
        return str(self.message.get("content") or "")

    @property
    def reasoning(self) -> str:
        message = self.message
        return str(message.get("reasoning_content") or message.get("reasoning") or "")

    @property
    def tool_calls(self) -> list[ToolCall]:
        calls = self.message.get("tool_calls") or []
        return [ToolCall.from_response(_mapping(item), index=i) for i, item in enumerate(calls)]

    @property
    def finish_reason(self) -> str | None:
        choices = self.choices
        return choices[0].get("finish_reason") if choices else None

    @property
    def usage(self) -> JSON:
        return dict(_mapping(self.raw.get("usage")))

    @property
    def id(self) -> str | None:
        value = self.raw.get("id")
        return str(value) if value is not None else None

    @property
    def model(self) -> str | None:
        value = self.raw.get("model")
        return str(value) if value is not None else None

    def model_dump(self) -> JSON:
        return dict(self.raw)


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One decoded server-sent event from a chat completion stream."""

    type: str
    data: JSON
    text: str = ""
    reasoning: str = ""
    tool_calls: tuple[ToolCall, ...] = ()

    @classmethod
    def from_chunk(cls, value: JSON) -> "StreamEvent":
        choices = value.get("choices") or []
        choice = _mapping(choices[0]) if choices else {}
        delta = _mapping(choice.get("delta"))
        text = str(delta.get("content") or "")
        reasoning = str(delta.get("reasoning_content") or delta.get("reasoning") or "")
        calls_list: list[ToolCall] = []
        for i, value in enumerate(delta.get("tool_calls") or []):
            item = _mapping(value)
            function = _mapping(item.get("function"))
            calls_list.append(
                ToolCall(
                    id=str(item.get("id") or ""),
                    name=str(function.get("name") or ""),
                    # Stream fields are fragments; an absent fragment must be empty,
                    # not the complete-message default of "{}".
                    arguments=str(function.get("arguments") or ""),
                    index=int(item.get("index", i)),
                )
            )
        calls = tuple(calls_list)
        if text:
            event_type = "text"
        elif reasoning:
            event_type = "reasoning"
        elif calls:
            event_type = "tool_call"
        elif choice.get("finish_reason"):
            event_type = "finish"
        else:
            event_type = "chunk"
        return cls(event_type, value, text, reasoning, calls)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    raw: JSON

    @property
    def id(self) -> str | None:
        value = self.raw.get("id")
        return str(value) if value is not None else None

    @property
    def supports_tools(self) -> bool:
        return bool(self.raw.get("supports_tools", False))

    @property
    def tool_protocol(self) -> str | None:
        value = self.raw.get("tool_protocol")
        return str(value) if value is not None else None

    def __getattr__(self, name: str) -> Any:
        # Keep runtime-specific metadata available without a rigid SDK schema.
        try:
            return self.raw[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True, slots=True)
class TokenizedInput:
    token_count: int
    tokens: tuple[int, ...] = ()
    raw: JSON = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenizeResponse:
    data: tuple[TokenizedInput, ...]
    raw: JSON = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(item.token_count for item in self.data)

    @property
    def token_counts(self) -> list[int]:
        return [item.token_count for item in self.data]


@dataclass(frozen=True, slots=True)
class ChatContext:
    raw: JSON

    @property
    def prompt_tokens(self) -> int | None:
        value = self.raw.get("prompt_tokens")
        return int(value) if value is not None else None

    @property
    def fits(self) -> bool | None:
        value = self.raw.get("fits")
        return bool(value) if value is not None else None

    @property
    def context_window(self) -> int | None:
        value = self.raw.get("context_window")
        return int(value) if value is not None else None

    @property
    def remaining_tokens(self) -> int | None:
        value = self.raw.get("remaining_tokens")
        return int(value) if value is not None else None

    def __getattr__(self, name: str) -> Any:
        try:
            return self.raw[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True, slots=True)
class ConversationInfo:
    raw: JSON

    @property
    def id(self) -> str:
        value = self.raw.get("id")
        if value is None:
            raise ValueError("Conversation response did not contain an id")
        return str(value)

    @property
    def title(self) -> str | None:
        value = self.raw.get("title")
        return str(value) if value is not None else None

    @property
    def messages(self) -> list[JSON]:
        return [dict(_mapping(item)) for item in (self.raw.get("messages") or [])]


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    raw: JSON

    @property
    def data(self) -> list[JSON]:
        return [dict(_mapping(item)) for item in (self.raw.get("data") or [])]

    @property
    def usage(self) -> JSON:
        return dict(_mapping(self.raw.get("usage")))


@dataclass(frozen=True, slots=True)
class AgentEvent:
    type: str
    data: JSON = field(default_factory=dict)

    @property
    def name(self) -> str | None:
        value = self.data.get("name")
        return str(value) if value is not None else None


@dataclass(frozen=True, slots=True)
class AgentTrace:
    events: tuple[AgentEvent, ...] = ()
    duration_ms: float = 0.0

    def tool_summary(self) -> list[JSON]:
        started = {
            str(event.data.get("call_id")): event.data
            for event in self.events
            if event.type == "tool_call_started"
        }
        summary: list[JSON] = []
        for event in self.events:
            if event.type != "tool_call_completed":
                continue
            data = event.data
            previous = started.get(str(data.get("call_id")), {})
            summary.append({
                "name": data.get("name", previous.get("name")),
                "arguments": previous.get("arguments", {}),
                "result": data.get("result"),
                "duration_ms": data.get("duration_ms", 0.0),
                "success": data.get("success", True),
                "error": data.get("error"),
            })
        return summary


@dataclass(frozen=True, slots=True)
class AgentResult:
    text: str
    response: ChatResponse
    conversation_id: str | None = None
    trace: AgentTrace | None = None
    parsed: Any | None = None

    @property
    def reasoning(self) -> str:
        return self.response.reasoning

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.response.tool_calls

    def __getattr__(self, name: str) -> Any:
        # Structured-output fields remain conveniently accessible on the result.
        parsed = object.__getattribute__(self, "parsed")
        if parsed is not None and hasattr(parsed, name):
            return getattr(parsed, name)
        raise AttributeError(name)
