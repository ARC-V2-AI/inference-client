from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from .errors import APIError


class SyncTransport:
    def __init__(self, base_url: str, api_key: str | None, timeout: float):
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.client.request(method, path, **kwargs)
        return _parse_response(response)

    def stream_sse(self, method: str, path: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        with self.client.stream(method, path, **kwargs) as response:
            if response.status_code >= 400:
                body = response.read()
                raise _api_error(response.status_code, body)
            for line in response.iter_lines():
                event = _parse_sse_line(line)
                if event is not None:
                    yield event


class AsyncTransport:
    def __init__(self, base_url: str, api_key: str | None, timeout: float):
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    async def close(self) -> None:
        await self.client.aclose()

    async def request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self.client.request(method, path, **kwargs)
        return _parse_response(response)

    async def stream_sse(self, method: str, path: str, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        async with self.client.stream(method, path, **kwargs) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise _api_error(response.status_code, body)
            async for line in response.aiter_lines():
                event = _parse_sse_line(line)
                if event is not None:
                    yield event


def _parse_response(response: httpx.Response) -> Any:
    if response.status_code >= 400:
        raise _api_error(response.status_code, response.content)
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return response.text


def _api_error(status: int, body: bytes) -> APIError:
    try:
        parsed = json.loads(body.decode("utf-8"))
        detail = parsed.get("detail", parsed)
    except Exception:
        detail = body.decode("utf-8", errors="replace")
    return APIError(status, str(detail), detail)


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return {"__done__": True}
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {"__raw__": data}
