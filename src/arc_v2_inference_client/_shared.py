from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .types import ChatResponse, StreamEvent, TokenizeResponse, TokenizedInput


def normalize_base_url(base_url: str) -> str:
    """Normalize root or `/v1` URLs to a trailing-slash `/v1/` base URL."""
    value = base_url.strip().rstrip("/")
    if not value:
        raise ValueError("base_url must not be empty")
    parts = urlsplit(value)
    path = parts.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path + "/", parts.query, parts.fragment))


def chat_payload(
    messages: Sequence[Any],
    *,
    model: str | None = None,
    stream: bool = False,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    stop: list[str] | None = None,
    response_format: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    tools: Sequence[Any] | None = None,
    tool_choice: Any = None,
    reasoning_effort: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [message_dict(item) for item in messages],
        "stream": stream,
    }
    values = {
        "model": model,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stop": stop,
        "response_format": response_format,
        "metadata": metadata,
        "tools": [tool_dict(item) for item in tools] if tools is not None else None,
        "tool_choice": tool_choice,
        "reasoning_effort": reasoning_effort,
    }
    payload.update({key: value for key, value in values.items() if value is not None})
    payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


def message_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, Mapping):
        return dict(message)
    method = getattr(message, "to_dict", None)
    if callable(method):
        return dict(method())
    method = getattr(message, "model_dump", None)
    if callable(method):
        return dict(method(exclude_none=True))
    raise TypeError(f"Unsupported message type: {type(message).__name__}")


def tool_dict(tool: Any) -> dict[str, Any]:
    if isinstance(tool, Mapping):
        return dict(tool)
    method = getattr(tool, "to_dict", None)
    if callable(method):
        return dict(method())
    method = getattr(tool, "definition", None)
    if callable(method):
        return dict(method())
    definition = getattr(tool, "__arc_v2_tool__", None)
    if isinstance(definition, Mapping):
        return dict(definition)
    method = getattr(definition, "definition", None)
    if callable(method):
        return dict(method())
    raise TypeError(f"Unsupported tool definition: {type(tool).__name__}")


def response_json(response: Any, method: str, path: str) -> dict[str, Any]:
    if response.is_success:
        if response.status_code == 204 or not response.content:
            return {}
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"{method} {path} returned a non-object JSON response")
        return data
    detail: Any = response.text
    try:
        body = response.json()
        if isinstance(body, dict):
            detail = body.get("detail", body.get("error", body))
    except (ValueError, json.JSONDecodeError):
        pass
    from .errors import InferenceAPIError
    raise InferenceAPIError(response.status_code, method, path, detail)


def parse_sse_data(data: str) -> dict[str, Any] | None:
    if data.strip() == "[DONE]":
        return None
    try:
        value = json.loads(data)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def tokenize_response(raw: dict[str, Any]) -> TokenizeResponse:
    values = raw.get("data") or []
    items: list[TokenizedInput] = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        tokens = value.get("tokens") or []
        items.append(
            TokenizedInput(
                token_count=int(value.get("token_count", len(tokens))),
                tokens=tuple(int(token) for token in tokens),
                raw=dict(value),
            )
        )
    return TokenizeResponse(tuple(items), raw)


def stream_event(raw: dict[str, Any]) -> StreamEvent:
    return StreamEvent.from_chunk(raw)
