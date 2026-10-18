from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping, Sequence
from typing import Any

import httpx

from ._shared import (
    chat_payload,
    parse_sse_data,
    response_json,
    stream_event,
    tokenize_response,
)
from .types import (
    ChatContext,
    ChatResponse,
    ConversationInfo,
    EmbeddingResponse,
    ModelInfo,
    StreamEvent,
)

_DEFAULT_URL = "http://127.0.0.1:7842"


class InferenceClient:
    """Synchronous client for the ARC V2 inference-runtime HTTP API.

    `base_url` may be the server root (`http://127.0.0.1:7842`) or the
    versioned API root (`http://127.0.0.1:7842/v1`).
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_URL,
        *,
        api_key: str | None = None,
        timeout: float | httpx.Timeout = 120.0,
        headers: Mapping[str, str] | None = None,
        verify: bool | str = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        from ._shared import normalize_base_url

        request_headers = {"Accept": "application/json"}
        if api_key:
            request_headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            request_headers.update(headers)
        self.base_url = normalize_base_url(base_url)
        self._http = httpx.Client(
            base_url=self.base_url,
            headers=request_headers,
            timeout=timeout,
            verify=verify,
            transport=transport,
        )
        self.chat = _ChatResource(self)
        self.completions = _CompletionResource(self)
        self.embeddings = _EmbeddingResource(self)
        self.models = _ModelsResource(self)
        self.model = _ModelResource(self)
        self.tokenizer = _TokenizerResource(self)
        self.context = _ContextResource(self)
        self.conversations = _ConversationsResource(self)

    def request(self, method: str, path: str, *, json_body: Any = None, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        response = self._http.request(method, path.lstrip("/"), json=json_body, params=params)
        return response_json(response, method.upper(), path)

    def _stream(self, path: str, payload: dict[str, Any]) -> Iterator[StreamEvent]:
        with self._http.stream(
            "POST",
            path.lstrip("/"),
            json=payload,
            headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"},
        ) as response:
            if not response.is_success:
                response.read()
                response_json(response, "POST", path)
            data_lines: list[str] = []
            for raw_line in response.iter_lines():
                line = raw_line.strip()
                if not line:
                    if data_lines:
                        data = parse_sse_data("\n".join(data_lines))
                        data_lines.clear()
                        if data is None:
                            return
                        if data:
                            yield stream_event(data)
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if data_lines:
                data = parse_sse_data("\n".join(data_lines))
                if data:
                    yield stream_event(data)

    def tokenize(
        self,
        input: str | Sequence[str],
        *,
        add_bos: bool = False,
        special: bool = True,
        include_tokens: bool = False,
    ):
        return self.tokenizer.tokenize(
            input, add_bos=add_bos, special=special, include_tokens=include_tokens
        )

    def conversation(self, *, system: str | None = None, title: str | None = None):
        """Create a persisted server-side conversation, optionally with system instructions."""
        return self.conversations.create(title=title, system=system)

    def agent(self, **kwargs: Any):
        from .agent import Agent
        return Agent(self, **kwargs)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "InferenceClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class AsyncInferenceClient:
    """Asynchronous client for the ARC V2 inference-runtime HTTP API."""

    def __init__(
        self,
        base_url: str = _DEFAULT_URL,
        *,
        api_key: str | None = None,
        timeout: float | httpx.Timeout = 120.0,
        headers: Mapping[str, str] | None = None,
        verify: bool | str = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        from ._shared import normalize_base_url

        request_headers = {"Accept": "application/json"}
        if api_key:
            request_headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            request_headers.update(headers)
        self.base_url = normalize_base_url(base_url)
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            headers=request_headers,
            timeout=timeout,
            verify=verify,
            transport=transport,
        )
        self.chat = _AsyncChatResource(self)
        self.completions = _AsyncCompletionResource(self)
        self.embeddings = _AsyncEmbeddingResource(self)
        self.models = _AsyncModelsResource(self)
        self.model = _AsyncModelResource(self)
        self.tokenizer = _AsyncTokenizerResource(self)
        self.context = _AsyncContextResource(self)
        self.conversations = _AsyncConversationsResource(self)

    async def request(self, method: str, path: str, *, json_body: Any = None, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        response = await self._http.request(method, path.lstrip("/"), json=json_body, params=params)
        return response_json(response, method.upper(), path)

    async def _stream(self, path: str, payload: dict[str, Any]) -> AsyncIterator[StreamEvent]:
        async with self._http.stream(
            "POST",
            path.lstrip("/"),
            json=payload,
            headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"},
        ) as response:
            if not response.is_success:
                await response.aread()
                response_json(response, "POST", path)
            data_lines: list[str] = []
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line:
                    if data_lines:
                        data = parse_sse_data("\n".join(data_lines))
                        data_lines.clear()
                        if data is None:
                            return
                        if data:
                            yield stream_event(data)
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if data_lines:
                data = parse_sse_data("\n".join(data_lines))
                if data:
                    yield stream_event(data)

    async def tokenize(
        self,
        input: str | Sequence[str],
        *,
        add_bos: bool = False,
        special: bool = True,
        include_tokens: bool = False,
    ):
        return await self.tokenizer.tokenize(
            input, add_bos=add_bos, special=special, include_tokens=include_tokens
        )

    async def conversation(self, *, system: str | None = None, title: str | None = None):
        return await self.conversations.create(title=title, system=system)

    def agent(self, **kwargs: Any):
        from .agent import AsyncAgent
        return AsyncAgent(self, **kwargs)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "AsyncInferenceClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()


class _ChatResource:
    def __init__(self, client: InferenceClient) -> None:
        self.client = client
        self.completions = self

    def create(self, messages: Iterable[Any], **kwargs: Any):
        payload = chat_payload(list(messages), **kwargs)
        if payload["stream"]:
            return self.client._stream("chat/completions", payload)
        return ChatResponse(self.client.request("POST", "chat/completions", json_body=payload))

    def context(self, messages: Iterable[Any], **kwargs: Any) -> ChatContext:
        payload = chat_payload(list(messages), stream=kwargs.pop("stream", False), **kwargs)
        return self.client.context.check(payload=payload)


class _AsyncChatResource:
    def __init__(self, client: AsyncInferenceClient) -> None:
        self.client = client
        self.completions = self

    async def create(self, messages: Iterable[Any], **kwargs: Any):
        payload = chat_payload(list(messages), **kwargs)
        if payload["stream"]:
            return self.client._stream("chat/completions", payload)
        return ChatResponse(await self.client.request("POST", "chat/completions", json_body=payload))

    async def context(self, messages: Iterable[Any], **kwargs: Any) -> ChatContext:
        payload = chat_payload(list(messages), stream=kwargs.pop("stream", False), **kwargs)
        return await self.client.context.check(payload=payload)


class _CompletionResource:
    def __init__(self, client: InferenceClient) -> None:
        self.client = client

    def create(self, prompt: str | Sequence[str], **kwargs: Any) -> dict[str, Any] | Iterator[StreamEvent]:
        payload = {"prompt": prompt}
        payload.update({key: value for key, value in kwargs.items() if value is not None})
        if payload.get("stream"):
            return self.client._stream("completions", payload)
        return self.client.request("POST", "completions", json_body=payload)


class _AsyncCompletionResource:
    def __init__(self, client: AsyncInferenceClient) -> None:
        self.client = client

    async def create(self, prompt: str | Sequence[str], **kwargs: Any):
        payload = {"prompt": prompt}
        payload.update({key: value for key, value in kwargs.items() if value is not None})
        if payload.get("stream"):
            return self.client._stream("completions", payload)
        return await self.client.request("POST", "completions", json_body=payload)


class _EmbeddingResource:
    def __init__(self, client: InferenceClient) -> None:
        self.client = client

    def create(self, input: str | Sequence[str], *, input_type: str = "document", instruction: str | None = None, **kwargs: Any) -> EmbeddingResponse:
        payload: dict[str, Any] = {"input": input, "input_type": input_type}
        if instruction is not None:
            payload["instruction"] = instruction
        payload.update({key: value for key, value in kwargs.items() if value is not None})
        return EmbeddingResponse(self.client.request("POST", "embeddings", json_body=payload))


class _AsyncEmbeddingResource:
    def __init__(self, client: AsyncInferenceClient) -> None:
        self.client = client

    async def create(self, input: str | Sequence[str], *, input_type: str = "document", instruction: str | None = None, **kwargs: Any) -> EmbeddingResponse:
        payload: dict[str, Any] = {"input": input, "input_type": input_type}
        if instruction is not None:
            payload["instruction"] = instruction
        payload.update({key: value for key, value in kwargs.items() if value is not None})
        return EmbeddingResponse(await self.client.request("POST", "embeddings", json_body=payload))


class _ModelResource:
    def __init__(self, client: InferenceClient) -> None:
        self.client = client

    def get(self) -> ModelInfo:
        return ModelInfo(self.client.request("GET", "model"))


class _AsyncModelResource:
    def __init__(self, client: AsyncInferenceClient) -> None:
        self.client = client

    async def get(self) -> ModelInfo:
        return ModelInfo(await self.client.request("GET", "model"))


class _ModelsResource:
    def __init__(self, client: InferenceClient) -> None:
        self.client = client

    def list(self) -> dict[str, Any]:
        return self.client.request("GET", "models")

    def current(self) -> ModelInfo:
        return self.client.model.get()


class _AsyncModelsResource:
    def __init__(self, client: AsyncInferenceClient) -> None:
        self.client = client

    async def list(self) -> dict[str, Any]:
        return await self.client.request("GET", "models")

    async def current(self) -> ModelInfo:
        return await self.client.model.get()


class _TokenizerResource:
    def __init__(self, client: InferenceClient) -> None:
        self.client = client

    def tokenize(self, input: str | Sequence[str], *, add_bos: bool = False, special: bool = True, include_tokens: bool = False):
        raw = self.client.request(
            "POST", "tokenize",
            json_body={"input": input, "add_bos": add_bos, "special": special, "include_tokens": include_tokens},
        )
        return tokenize_response(raw)


class _AsyncTokenizerResource:
    def __init__(self, client: AsyncInferenceClient) -> None:
        self.client = client

    async def tokenize(self, input: str | Sequence[str], *, add_bos: bool = False, special: bool = True, include_tokens: bool = False):
        raw = await self.client.request(
            "POST", "tokenize",
            json_body={"input": input, "add_bos": add_bos, "special": special, "include_tokens": include_tokens},
        )
        return tokenize_response(raw)


class _ContextResource:
    def __init__(self, client: InferenceClient) -> None:
        self.client = client

    def check(self, messages: Iterable[Any] | None = None, *, payload: dict[str, Any] | None = None, **kwargs: Any) -> ChatContext:
        if payload is None:
            if messages is None:
                raise TypeError("messages or payload is required")
            payload = chat_payload(list(messages), **kwargs)
        return ChatContext(self.client.request("POST", "chat/context", json_body=payload))


class _AsyncContextResource:
    def __init__(self, client: AsyncInferenceClient) -> None:
        self.client = client

    async def check(self, messages: Iterable[Any] | None = None, *, payload: dict[str, Any] | None = None, **kwargs: Any) -> ChatContext:
        if payload is None:
            if messages is None:
                raise TypeError("messages or payload is required")
            payload = chat_payload(list(messages), **kwargs)
        return ChatContext(await self.client.request("POST", "chat/context", json_body=payload))


class _ConversationsResource:
    def __init__(self, client: InferenceClient) -> None:
        self.client = client

    def create(self, *, title: str | None = None, system: str | None = None):
        payload = {"title": title} if title is not None else {}
        raw = self.client.request("POST", "conversations", json_body=payload)
        from .conversation import Conversation
        return Conversation(self.client, ConversationInfo(raw), system=system)

    def list(self) -> list[ConversationInfo]:
        raw = self.client.request("GET", "conversations")
        values = raw.get("data", raw.get("conversations", []))
        return [ConversationInfo(dict(value)) for value in values if isinstance(value, Mapping)]

    def get(self, conversation_id: str):
        raw = self.client.request("GET", f"conversations/{conversation_id}")
        from .conversation import Conversation
        return Conversation(self.client, ConversationInfo(raw))

    def delete(self, conversation_id: str) -> bool:
        self.client.request("DELETE", f"conversations/{conversation_id}")
        return True


class _AsyncConversationsResource:
    def __init__(self, client: AsyncInferenceClient) -> None:
        self.client = client

    async def create(self, *, title: str | None = None, system: str | None = None):
        payload = {"title": title} if title is not None else {}
        raw = await self.client.request("POST", "conversations", json_body=payload)
        from .conversation import AsyncConversation
        return AsyncConversation(self.client, ConversationInfo(raw), system=system)

    async def list(self) -> list[ConversationInfo]:
        raw = await self.client.request("GET", "conversations")
        values = raw.get("data", raw.get("conversations", []))
        return [ConversationInfo(dict(value)) for value in values if isinstance(value, Mapping)]

    async def get(self, conversation_id: str):
        raw = await self.client.request("GET", f"conversations/{conversation_id}")
        from .conversation import AsyncConversation
        return AsyncConversation(self.client, ConversationInfo(raw))

    async def delete(self, conversation_id: str) -> bool:
        await self.client.request("DELETE", f"conversations/{conversation_id}")
        return True
