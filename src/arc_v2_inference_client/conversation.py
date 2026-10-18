from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from typing import Any

from ._shared import chat_payload
from .types import ChatResponse, ConversationInfo, StreamEvent


class Conversation:
    """A persisted conversation owned by the inference runtime.

    Only newly added messages are sent to the conversation completion endpoint.
    The server is authoritative for transcript persistence and context pruning.
    """

    def __init__(self, client: Any, info: ConversationInfo, *, system: str | None = None) -> None:
        self.client = client
        self.info = info
        self.id = info.id
        self._pending_system = system

    @property
    def title(self) -> str | None:
        return self.info.title

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.info.messages

    def refresh(self) -> ConversationInfo:
        self.info = ConversationInfo(self.client.request("GET", f"conversations/{self.id}"))
        return self.info

    def complete(self, messages: Iterable[Any], **kwargs: Any) -> ChatResponse:
        additions = list(messages)
        if self._pending_system is not None:
            additions.insert(0, {"role": "system", "content": self._pending_system})
        payload = chat_payload(additions, **kwargs)
        payload["stream"] = False
        response = ChatResponse(
            self.client.request(
                "POST", f"conversations/{self.id}/chat/completions", json_body=payload
            )
        )
        self._pending_system = None
        return response

    def send(self, prompt: str, **kwargs: Any) -> ChatResponse:
        return self.complete([{"role": "user", "content": prompt}], **kwargs)

    def stream_messages(self, messages: Iterable[Any], **kwargs: Any) -> Iterator[StreamEvent]:
        additions = list(messages)
        if self._pending_system is not None:
            additions.insert(0, {"role": "system", "content": self._pending_system})
        kwargs.pop("stream", None)
        payload = chat_payload(additions, stream=True, **kwargs)
        completed = False
        try:
            yield from self.client._stream(
                f"conversations/{self.id}/chat/completions", payload
            )
            completed = True
        finally:
            if completed:
                self._pending_system = None

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[StreamEvent]:
        return self.stream_messages([{"role": "user", "content": prompt}], **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "messages": self.messages}


class AsyncConversation:
    """Asynchronous counterpart to :class:`Conversation`."""

    def __init__(self, client: Any, info: ConversationInfo, *, system: str | None = None) -> None:
        self.client = client
        self.info = info
        self.id = info.id
        self._pending_system = system

    @property
    def title(self) -> str | None:
        return self.info.title

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.info.messages

    async def refresh(self) -> ConversationInfo:
        self.info = ConversationInfo(await self.client.request("GET", f"conversations/{self.id}"))
        return self.info

    async def complete(self, messages: Iterable[Any], **kwargs: Any) -> ChatResponse:
        additions = list(messages)
        if self._pending_system is not None:
            additions.insert(0, {"role": "system", "content": self._pending_system})
        payload = chat_payload(additions, **kwargs)
        payload["stream"] = False
        response = ChatResponse(
            await self.client.request(
                "POST", f"conversations/{self.id}/chat/completions", json_body=payload
            )
        )
        self._pending_system = None
        return response

    async def send(self, prompt: str, **kwargs: Any) -> ChatResponse:
        return await self.complete([{"role": "user", "content": prompt}], **kwargs)

    async def stream_messages(self, messages: Iterable[Any], **kwargs: Any) -> AsyncIterator[StreamEvent]:
        additions = list(messages)
        if self._pending_system is not None:
            additions.insert(0, {"role": "system", "content": self._pending_system})
        kwargs.pop("stream", None)
        payload = chat_payload(additions, stream=True, **kwargs)
        completed = False
        try:
            async for event in self.client._stream(
                f"conversations/{self.id}/chat/completions", payload
            ):
                yield event
            completed = True
        finally:
            if completed:
                self._pending_system = None

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        async for event in self.stream_messages([{"role": "user", "content": prompt}], **kwargs):
            yield event

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "messages": self.messages}
