from __future__ import annotations

from typing import Any


class InferenceClientError(RuntimeError):
    """Base error raised by the ARC V2 inference client."""


class InferenceAPIError(InferenceClientError):
    def __init__(self, status_code: int, method: str, path: str, detail: Any) -> None:
        self.status_code = status_code
        self.method = method
        self.path = path
        self.detail = detail
        super().__init__(f"{method} {path} returned HTTP {status_code}: {detail}")


class AgentLoopError(InferenceClientError):
    """Raised when the agent exceeds its configured tool-call iteration limit."""


class ToolExecutionError(InferenceClientError):
    """Raised when a local tool cannot be executed or its arguments are invalid."""
