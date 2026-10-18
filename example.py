#!/usr/bin/env python3
"""Interactive ARC V2 inference-runtime demo.

Run from the SDK repository:
    uv run example.py
    uv run example.py --demo
    uv run example.py --reasoning

Commands inside chat:
    /reasoning [on|off|toggle]  Enable/disable server reasoning effort and display
                                reasoning deltas when the server emits them.
    /clear                      Clear local chat history.
    /tools                      Show available local tools.
    /help                       Show commands.
    /quit                       Exit.

The demo uses tool_choice="auto". The runtime logs supplied with this example
show that forcing a named tool can currently produce an incomplete Qwen XML
function header and an empty generation, which is a server-side generation /
parser issue. Auto mode exercises the same tool-call API without triggering
that specific forced-tool path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

from arc_v2_inference_client import InferenceClient

DEFAULT_URL = "http://127.0.0.1:7842"
SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Use the available tools when they are "
    "useful or explicitly requested. After a tool returns, use its result to "
    "answer the user. Do not claim a tool ran unless its result is present."
)
MAX_OUTPUT_TOKENS = 512
MAX_TOOL_ROUNDS = 6

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Return the local time on the machine running this example.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


_COLOR = _supports_color()


def style(value: str, code: str) -> str:
    return f"\033[{code}m{value}\033[0m" if _COLOR else value


def tool_result(name: str, arguments: dict[str, Any]) -> str:
    """Execute the safe local demonstration tools and return JSON for the model."""
    if name == "multiply":
        a, b = arguments.get("a"), arguments.get("b")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (a, b)):
            raise ValueError("arguments 'a' and 'b' must be numbers")
        result: dict[str, Any] = {"a": a, "b": b, "result": a * b}
    elif name == "get_current_time":
        result = {"local_time": datetime.now().astimezone().isoformat(timespec="seconds")}
    else:
        raise ValueError(f"Unknown tool: {name}")
    return json.dumps(result, ensure_ascii=False)


def print_banner(url: str, model: Any, reasoning: bool) -> None:
    title = "ARC V2  /  INFERENCE CLIENT"
    print(style("┌" + "─" * 66 + "┐", "2"))
    print(style("│ " + title.ljust(65) + "│", "1;32"))
    print(style("├" + "─" * 66 + "┤", "2"))
    print(f"│ {'Runtime':<9} {url[:54]:<54} │")
    print(f"│ {'Model':<9} {str(model.id or 'unknown')[:54]:<54} │")
    tool_status = 'available' if model.supports_tools else 'not advertised'
    print(f"│ {'Tools':<9} {tool_status:<54} │")
    reasoning_status = 'on · effort=on' if reasoning else 'off · effort=none'
    print(f"│ {'Reasoning':<9} {reasoning_status:<54} │")
    print(style("└" + "─" * 66 + "┘", "2"))
    print("Enter a message, or type /help for commands.\n")


def show_tools() -> None:
    print(style("Available local tools", "1;36"))
    print("  multiply(a, b)       Multiply two numbers.")
    print("  get_current_time()   Read this machine's local time.")
    print("These functions run in this example process; the model only requests them.\n")


def count_tokens(
    client: InferenceClient,
    prompt: str,
    messages: list[dict[str, Any]],
    reasoning: bool,
) -> None:
    """Print raw user-input tokens plus exact rendered prompt accounting."""
    if prompt:
        tokenized = client.tokenizer.tokenize(prompt, include_tokens=False)
        count = tokenized.data[0].token_count if tokenized.data else "?"
    else:
        count = "n/a"
    effort = "on" if reasoning else "none"
    context = client.chat.context(
        messages,
        tools=TOOLS,
        tool_choice="auto",
        max_output_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort=effort,
    )
    context_size = context.context_window if context.context_window is not None else "?"
    remaining = context.remaining_tokens if context.remaining_tokens is not None else "?"
    fits = "yes" if context.fits else "no" if context.fits is False else "unknown"
    print(style("  TOKENS", "1;35") +
          f"  input={count}  prompt={context.prompt_tokens}  "
          f"remaining={remaining}/{context_size}  fits={fits}  reasoning={effort}")


def stream_model_turn(
    client: InferenceClient,
    messages: list[dict[str, Any]],
    *,
    reasoning: bool,
) -> tuple[str, list[dict[str, str]], str | None]:
    """Stream one model response and collect incremental native tool-call deltas."""
    effort = "on" if reasoning else "none"
    stream = client.chat.create(
        messages,
        tools=TOOLS,
        tool_choice="auto",
        stream=True,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,
        reasoning_effort=effort,
    )

    text_parts: list[str] = []
    tool_acc: dict[int, dict[str, str]] = {}
    finish_reason: str | None = None
    mode: str | None = None

    for event in stream:
        choices = event.data.get("choices") or []
        if choices:
            finish_reason = choices[0].get("finish_reason") or finish_reason

        if event.reasoning and reasoning:
            if mode != "reasoning":
                if mode is not None:
                    print()
                print(style("  THINK │ ", "2;35"), end="", flush=True)
                mode = "reasoning"
            print(style(event.reasoning, "2;35"), end="", flush=True)

        if event.text:
            if mode != "answer":
                if mode is not None:
                    print()
                print(style("  ARC  │ ", "1;32"), end="", flush=True)
                mode = "answer"
            print(event.text, end="", flush=True)
            text_parts.append(event.text)

        for call in event.tool_calls:
            item = tool_acc.setdefault(
                call.index,
                {"id": "", "name": "", "arguments": ""},
            )
            if call.id:
                item["id"] = call.id
            if call.name:
                item["name"] += call.name
            if call.arguments:
                item["arguments"] += call.arguments

    if mode is not None:
        print()

    calls: list[dict[str, str]] = []
    for index in sorted(tool_acc):
        item = tool_acc[index]
        if item["name"]:
            calls.append(item)
    return "".join(text_parts).strip(), calls, finish_reason


def chat_turn(
    client: InferenceClient,
    messages: list[dict[str, Any]],
    prompt: str,
    *,
    reasoning: bool,
) -> int:
    """Append a user turn, stream model responses, execute tools, and continue."""
    messages.append({"role": "user", "content": prompt})
    executed = 0

    for round_number in range(1, MAX_TOOL_ROUNDS + 1):
        count_tokens(client, prompt if round_number == 1 else "", messages, reasoning)
        answer, calls, finish_reason = stream_model_turn(client, messages, reasoning=reasoning)

        if not calls:
            messages.append({"role": "assistant", "content": answer})
            if not answer:
                print(style(f"  [no assistant text; finish_reason={finish_reason!r}]", "1;33"))
            return executed

        # Keep the original tool-call message in protocol-compliant OpenAI format.
        raw_calls: list[dict[str, Any]] = []
        for index, call in enumerate(calls):
            call_id = call["id"] or f"call_local_{round_number}_{index}"
            raw_calls.append({
                "id": call_id,
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"] or "{}"},
            })
        messages.append({"role": "assistant", "content": answer or None, "tool_calls": raw_calls})

        for call, raw_call in zip(calls, raw_calls):
            name = call["name"]
            raw_arguments = call["arguments"] or "{}"
            try:
                parsed = json.loads(raw_arguments)
                if not isinstance(parsed, dict):
                    raise ValueError("tool arguments must be a JSON object")
                print(style(f"  TOOL │ {name}", "1;36") + " " + json.dumps(parsed, ensure_ascii=False))
                result = tool_result(name, parsed)
                print(style("  RESULT │ ", "36") + result)
            except Exception as exc:
                result = json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
                print(style("  TOOL ERROR │ ", "1;31") + result)

            messages.append({
                "role": "tool",
                "tool_call_id": raw_call["id"],
                "name": name,
                "content": result,
            })
            executed += 1

    raise RuntimeError(f"Model exceeded {MAX_TOOL_ROUNDS} tool-call rounds")


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive streaming ARC V2 inference demo")
    parser.add_argument("--url", default=DEFAULT_URL, help="Runtime server URL")
    parser.add_argument("--reasoning", action="store_true", help="Start with reasoning enabled (reasoning_effort=on)")
    parser.add_argument("--demo", action="store_true", help="Run a tool-call smoke test and exit")
    args = parser.parse_args()

    reasoning = bool(args.reasoning)
    with InferenceClient(args.url, timeout=180) as client:
        model = client.model.get()
        print_banner(args.url, model, reasoning)
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if args.demo:
            prompt = (
                "You must use the multiply tool now with a=19 and b=23. "
                "After the tool returns, state the result in one short sentence."
            )
            print(style("  DEMO │ Requesting a tool via tool_choice=auto", "1;34"))
            count = chat_turn(client, messages, prompt, reasoning=reasoning)
            answer = messages[-1].get("content") if messages and messages[-1].get("role") == "assistant" else ""
            # Require both a completed tool round trip and a final natural-language response.
            has_tool_result = any(m.get("role") == "tool" for m in messages)
            if not has_tool_result or not isinstance(answer, str) or not answer.strip():
                print(style("\nDEMO FAILED: No tool round trip and final answer were both observed.", "1;31"))
                print("The demo intentionally uses auto tool choice. Do not force a named tool until the server's")
                print("Qwen XML parser / constrained-generation path is fixed; see the server log message above.")
                return 1
            print(style(f"\nDEMO PASSED │ {count} local tool call(s), final answer received", "1;32"))
            return 0

        show_tools()
        while True:
            try:
                prompt = input(style("You  › ", "1;34")).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n" + style("Session ended.", "2"))
                break

            if not prompt:
                continue
            command, _, remainder = prompt.partition(" ")
            command = command.lower()

            if command in {"/quit", "/exit"}:
                print(style("Session ended.", "2"))
                break
            if command == "/help":
                print("  /reasoning [on|off|toggle]  Enable or disable server reasoning.")
                print("  /clear                      Clear local conversation history.")
                print("  /tools                      List available local tools.")
                print("  /quit                       Exit.\n")
                continue
            if command == "/tools":
                show_tools()
                continue
            if command == "/clear":
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                print(style("History cleared.\n", "2"))
                continue
            if command == "/reasoning":
                choice = remainder.strip().lower() or "toggle"
                if choice in {"toggle", "on"}:
                    reasoning = True if choice == "on" else not reasoning
                elif choice == "off":
                    reasoning = False
                else:
                    print("Usage: /reasoning [on|off|toggle]\n")
                    continue
                state = "ON (reasoning_effort=on; show reasoning deltas when emitted)" if reasoning else "OFF (none)"
                print(style(f"Reasoning: {state}\n", "1;35"))
                continue
            if command.startswith("/"):
                print(f"Unknown command: {command}. Type /help.\n")
                continue

            try:
                tools_executed = chat_turn(client, messages, prompt, reasoning=reasoning)
                if tools_executed:
                    print(style(f"  ✓ executed {tools_executed} local tool call(s)", "2;36"))
                print()
            except KeyboardInterrupt:
                print("\nTurn interrupted. You can continue or use /clear if history is inconsistent.\n")
            except Exception as exc:
                print(style(f"\nRequest failed: {exc}\n", "1;31"), file=sys.stderr)
                print("The session remains available; use /clear if you want to reset history.\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
