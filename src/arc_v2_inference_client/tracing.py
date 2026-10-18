from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass(slots=True)
class TraceEvent:
    """One observable step in an agent execution."""

    type: str
    timestamp: float = field(default_factory=time.time)
    iteration: int | None = None
    name: str | None = None
    call_id: str | None = None
    arguments: Any = None
    result: Any = None
    duration_ms: float | None = None
    success: bool | None = None
    error: str | None = None
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)


@dataclass(slots=True)
class ToolTrace:
    call_id: str
    name: str
    arguments: Any
    result: Any = None
    success: bool = True
    error: str | None = None
    duration_ms: float = 0.0
    iteration: int | None = None


@dataclass(slots=True)
class AgentTrace:
    """Complete observable execution trace for one agent run."""

    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    events: list[TraceEvent] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return (end - self.started_at) * 1000

    @property
    def tool_calls(self) -> list[TraceEvent]:
        return [event for event in self.events if event.type == "tool_call_completed"]

    @property
    def iterations(self) -> int:
        return sum(1 for event in self.events if event.type == "iteration_started")

    @property
    def errors(self) -> list[TraceEvent]:
        return [event for event in self.events if event.type == "error" or event.success is False]

    def add(self, event_type: str, **kwargs: Any) -> TraceEvent:
        event = TraceEvent(type=event_type, **kwargs)
        self.events.append(event)
        return event

    def finish(self) -> "AgentTrace":
        self.finished_at = time.time()
        return self

    def tool_summary(self) -> list[ToolTrace]:
        summaries: list[ToolTrace] = []
        for event in self.tool_calls:
            summaries.append(
                ToolTrace(
                    call_id=event.call_id or "",
                    name=event.name or "",
                    arguments=event.arguments,
                    result=event.result,
                    success=event.success is not False,
                    error=event.error,
                    duration_ms=event.duration_ms or 0.0,
                    iteration=event.iteration,
                )
            )
        return summaries

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "iterations": self.iterations,
            "events": [
                {
                    "type": event.type,
                    "timestamp": event.timestamp,
                    "iteration": event.iteration,
                    "name": event.name,
                    "call_id": event.call_id,
                    "arguments": event.arguments,
                    "result": event.result,
                    "duration_ms": event.duration_ms,
                    "success": event.success,
                    "error": event.error,
                    "text": event.text,
                    "data": event.data,
                }
                for event in self.events
            ],
        }


@dataclass(slots=True)
class AgentRunResult:
    """Final agent response together with its execution trace."""

    response: Any
    trace: AgentTrace

    @property
    def text(self) -> str:
        return getattr(self.response, "text", str(self.response))

    @property
    def output(self) -> Any:
        return self.response


TraceCallback = Callable[[TraceEvent], None]
