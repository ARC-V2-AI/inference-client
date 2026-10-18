from __future__ import annotations

import inspect
import json
import types
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal, Union, get_args, get_origin, get_type_hints

from .errors import ToolExecutionError


ToolFunction = Callable[..., Any]


def _json_type(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Signature.empty or annotation is Any:
        return {}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        values = list(args)
        if values and all(isinstance(v, str) for v in values):
            return {"type": "string", "enum": values}
        if values and all(isinstance(v, bool) for v in values):
            return {"type": "boolean", "enum": values}
        if values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            return {"type": "integer", "enum": values}
        if values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            return {"type": "number", "enum": values}
        return {"enum": values}
    if origin in (Union, types.UnionType):
        return {"anyOf": [_json_type(value) for value in args]}
    if origin in (list, tuple, set, frozenset):
        item_type = args[0] if args else Any
        return {"type": "array", "items": _json_type(item_type)}
    if origin is dict:
        return {"type": "object", "additionalProperties": _json_type(args[1]) if len(args) > 1 else True}
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        values = [item.value for item in annotation]
        schema: dict[str, Any] = {"enum": values}
        if values and all(isinstance(value, str) for value in values):
            schema["type"] = "string"
        elif values and all(isinstance(value, bool) for value in values):
            schema["type"] = "boolean"
        elif values and all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            schema["type"] = "integer"
        elif values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            schema["type"] = "number"
        return schema
    if annotation is type(None):
        return {"type": "null"}
    if annotation is str:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is list:
        return {"type": "array", "items": {}}
    if annotation is dict:
        return {"type": "object", "additionalProperties": True}
    return {}


def _parameters_for(function: ToolFunction) -> dict[str, Any]:
    signature = inspect.signature(function)
    try:
        hints = get_type_hints(function)
    except (NameError, TypeError):
        hints = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if name in {"self", "cls"} or parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        annotation = hints.get(name, parameter.annotation)
        schema = _json_type(annotation)
        if parameter.default is not inspect.Signature.empty:
            # JSON Schema has no standard default semantics for function calling,
            # but preserving the default helps schema-aware clients.
            if parameter.default is None or isinstance(parameter.default, (str, int, float, bool, list, dict)):
                schema["default"] = parameter.default
        else:
            required.append(name)
        properties[name] = schema
    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        result["required"] = required
    return result


@dataclass(frozen=True, slots=True)
class Tool:
    function: ToolFunction
    name: str
    description: str
    parameters: dict[str, Any]

    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, arguments: str) -> str:
        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(f"Invalid JSON arguments for tool {self.name!r}") from exc
        if not isinstance(parsed, dict):
            raise ToolExecutionError(f"Arguments for tool {self.name!r} must be a JSON object")
        try:
            value = self.function(**parsed)
        except Exception as exc:
            raise ToolExecutionError(f"Tool {self.name!r} failed: {exc}") from exc
        if inspect.isawaitable(value):
            if inspect.iscoroutine(value):
                value.close()
            raise ToolExecutionError(
                f"Tool {self.name!r} is asynchronous; use AsyncAgent with AsyncInferenceClient"
            )
        return serialize_result(value)

    async def aexecute(self, arguments: str) -> str:
        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(f"Invalid JSON arguments for tool {self.name!r}") from exc
        if not isinstance(parsed, dict):
            raise ToolExecutionError(f"Arguments for tool {self.name!r} must be a JSON object")
        try:
            value = self.function(**parsed)
            if inspect.isawaitable(value):
                value = await value
        except Exception as exc:
            raise ToolExecutionError(f"Tool {self.name!r} failed: {exc}") from exc
        return serialize_result(value)


def ensure_tool(value: Any) -> Tool:
    if isinstance(value, Tool):
        return value
    if not callable(value):
        raise TypeError(f"Expected a callable or Tool, got {type(value).__name__}")
    existing = getattr(value, "__arc_v2_tool__", None)
    if isinstance(existing, Tool):
        return existing
    description = inspect.getdoc(value) or ""
    return Tool(value, value.__name__, description, _parameters_for(value))


def tool(
    function: ToolFunction | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
):
    """Mark a Python function as an inference tool.

    Supports both `@tool` and `@tool(name="lookup", description="...")`.
    Parameters are inferred from type annotations unless an explicit JSON Schema
    is supplied.
    """
    def decorate(fn: ToolFunction) -> ToolFunction:
        spec = Tool(
            function=fn,
            name=name or fn.__name__,
            description=description if description is not None else (inspect.getdoc(fn) or ""),
            parameters=parameters if parameters is not None else _parameters_for(fn),
        )
        setattr(fn, "__arc_v2_tool__", spec)
        # Retain the marker understood by earlier ARC inference SDK releases.
        setattr(fn, "__inference_tool__", spec)
        return fn

    if function is not None:
        if not callable(function):
            raise TypeError("@tool expects a function")
        return decorate(function)
    return decorate


def serialize_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)
