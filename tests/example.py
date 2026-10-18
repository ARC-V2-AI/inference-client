#!/usr/bin/env python3
"""End-to-end smoke test for the ARC V2 inference client and a live runtime.

Run from the client repository root:
    uv run example.py

Options:
    ARC_RUNTIME_URL=http://127.0.0.1:7842
    ARC_TEST_TIMEOUT_S=300
    uv run example.py --url http://127.0.0.1:7842 --timeout 300
    uv run example.py --skip-agent
    uv run example.py --embeddings  # requires a configured embedding model

This script sends real inference requests and creates temporary server-side
conversations. The conversations are deleted after their transcripts are checked.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
from typing import Any

from arc_v2_inference_client import AsyncInferenceClient, InferenceClient, tool


class SmokeTest:
    def __init__(self) -> None:
        self.passed = 0

    def check(self, condition: Any, message: str) -> None:
        if not condition:
            raise RuntimeError(f"FAIL: {message}")
        self.passed += 1
        print(f"PASS: {message}")


def marker(prefix: str = "ARC-E2E") -> str:
    return f"{prefix}-{secrets.token_hex(3).upper()}"


def run_sync_checks(
    test: SmokeTest,
    *,
    base_url: str,
    timeout: float,
    skip_agent: bool,
    test_embeddings: bool,
) -> None:
    print(f"\n=== Synchronous client: {base_url} ===")

    with InferenceClient(base_url, timeout=timeout) as client:
        # 1. Server/model metadata.
        model = client.model.get()
        test.check(bool(model.id), f"runtime is reachable; loaded model={model.id!r}")
        print(f"  tool protocol: {model.tool_protocol!r}; supports_tools={model.supports_tools}")

        # 2. Model-native tokenizer.
        tokenized = client.tokenize(
            ["ARC V2 inference client smoke test", "Tokenizer counts are model-specific."],
            include_tokens=False,
        )
        test.check(
            len(tokenized.data) == 2 and all(item.token_count > 0 for item in tokenized.data),
            f"tokenizer returns two positive counts: {tokenized.token_counts}",
        )

        # 3. Exact rendered prompt/context accounting on the server.
        context = client.chat.context(
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Reply with the single word READY."},
            ],
            max_output_tokens=64,
            reasoning_effort="none",
        )
        test.check(
            isinstance(context.prompt_tokens, int) and context.prompt_tokens > 0,
            f"context endpoint counts the rendered prompt ({context.prompt_tokens} tokens)",
        )
        test.check(context.fits is True, "small prompt fits the runtime context budget")

        # 4. A real, non-streaming stateless completion.
        direct_marker = marker("DIRECT")
        response = client.chat.create(
            [
                {"role": "system", "content": "Follow the user's output format exactly."},
                {
                    "role": "user",
                    "content": f"Return exactly this marker and nothing else: {direct_marker}",
                },
            ],
            max_output_tokens=64,
            temperature=0,
            reasoning_effort="none",
        )
        print(f"  stateless response: {response.text.strip()!r}")
        test.check(direct_marker.lower() in response.text.lower(), "stateless chat returns the requested marker")

        # 5. Real SSE streaming response.
        stream_marker = marker("STREAM")
        pieces: list[str] = []
        for event in client.chat.create(
            [{
                "role": "user",
                "content": f"Reply with this exact marker and nothing else: {stream_marker}",
            }],
            stream=True,
            max_output_tokens=64,
            temperature=0,
            reasoning_effort="none",
        ):
            if event.text:
                pieces.append(event.text)
                print(event.text, end="", flush=True)
        streamed_text = "".join(pieces)
        print()
        test.check(stream_marker.lower() in streamed_text.lower(), "SSE stream emits the requested marker")

        # 6. Server-persisted conversation: a second turn must recall the first turn.
        remembered = marker("MEMORY")
        conversation = client.conversation(
            title="ARC V2 client end-to-end smoke test",
            system="You are being tested. Follow exact-output requests carefully.",
        )
        conversation_id = conversation.id
        try:
            first = conversation.send(
                f"Remember this exact identifier for my next message: {remembered}. Reply only ACK.",
                max_output_tokens=64,
                temperature=0,
                reasoning_effort="none",
            )
            print(f"  conversation first turn: {first.text.strip()!r}")
            second = conversation.send(
                "What exact identifier did I ask you to remember? Reply with only the identifier.",
                max_output_tokens=64,
                temperature=0,
                reasoning_effort="none",
            )
            print(f"  conversation recall: {second.text.strip()!r}")
            test.check(
                remembered.lower() in second.text.lower(),
                "persisted conversation carries context into the next request",
            )

            saved = client.conversations.get(conversation_id)
            saved_json = json.dumps(saved.messages, ensure_ascii=False)
            roles = [message.get("role") for message in saved.messages]
            test.check(
                remembered in saved_json and roles.count("user") >= 2 and roles.count("assistant") >= 2,
                f"server transcript can be retrieved ({len(saved.messages)} messages; id={conversation_id})",
            )
        finally:
            try:
                client.conversations.delete(conversation_id)
                print("  temporary conversation deleted")
            except Exception as exc:  # cleanup must not hide the actual test failure
                print(f"  WARNING: could not delete conversation {conversation_id}: {exc}", file=sys.stderr)

        # 7. Optional embeddings; the runtime returns HTTP 503 when its embedding model is not configured.
        if test_embeddings:
            embeddings = client.embeddings.create(
                "ARC V2 local inference service", input_type="document"
            )
            vectors = [item.get("embedding") for item in embeddings.data]
            test.check(
                bool(vectors) and isinstance(vectors[0], list) and bool(vectors[0]),
                f"embedding endpoint returns a non-empty vector (dimensions={len(vectors[0]) if vectors else 0})",
            )
        else:
            print("SKIP: embeddings (pass --embeddings when a dedicated embedding model is configured)")

        # 8. End-to-end tool execution through the agent loop, when supported by the loaded model.
        if skip_agent:
            print("SKIP: agent/tool loop (--skip-agent)")
        elif not model.supports_tools:
            print("SKIP: agent/tool loop (runtime reports that the loaded model does not support tools)")
        else:
            expected_marker = marker("TOOL")
            tool_executions: list[dict[str, Any]] = []

            @tool(description="Record the supplied marker and integer in the local smoke-test harness.")
            def record_e2e_result(marker: str, value: int) -> dict[str, Any]:
                """Record a marker and integer to prove client-side tool execution."""
                result = {"marker": marker, "value": value, "ok": True}
                tool_executions.append(result)
                return result

            agent = client.agent(
                instructions="Use the provided tool exactly once when asked. After it returns, give a brief confirmation.",
                tools=[record_e2e_result],
                max_iterations=3,
                tool_choice={"type": "function", "function": {"name": "record_e2e_result"}},
                max_output_tokens=128,
                temperature=0,
                reasoning_effort="none",
                title="ARC V2 client agent end-to-end smoke test",
            )
            agent_conversation_id: str | None = None
            try:
                result = agent.run(
                    f"Call record_e2e_result exactly once with marker='{expected_marker}' and value=37. Then confirm completion.",
                    trace=True,
                )
                agent_conversation_id = result.conversation_id
                print(f"  agent response: {result.text.strip()!r}")
                test.check(
                    any(item.get("marker") == expected_marker and item.get("value") == 37 for item in tool_executions),
                    "agent received a tool call and executed the local Python function",
                )
                test.check(bool(result.text.strip()), "agent completes with a final assistant response")

                if agent_conversation_id:
                    transcript = client.conversations.get(agent_conversation_id)
                    roles = [message.get("role") for message in transcript.messages]
                    test.check("tool" in roles, f"agent tool result is persisted in the server transcript ({roles})")
            finally:
                if agent_conversation_id:
                    try:
                        client.conversations.delete(agent_conversation_id)
                        print("  temporary agent conversation deleted")
                    except Exception as exc:
                        print(f"  WARNING: could not delete agent conversation {agent_conversation_id}: {exc}", file=sys.stderr)


def run_async_checks(test: SmokeTest, *, base_url: str, timeout: float) -> None:
    async def _run() -> None:
        print(f"\n=== Asynchronous client: {base_url} ===")
        async with AsyncInferenceClient(base_url, timeout=timeout) as client:
            model = await client.model.get()
            test.check(bool(model.id), f"async client reads runtime model metadata ({model.id!r})")

            tokenized = await client.tokenize("Async tokenizer smoke test", include_tokens=False)
            test.check(
                len(tokenized.data) == 1 and tokenized.data[0].token_count > 0,
                f"async tokenizer returns a positive count ({tokenized.token_counts})",
            )

            async_marker = marker("ASYNC")
            response = await client.chat.create(
                [{"role": "user", "content": f"Reply with only this marker: {async_marker}"}],
                max_output_tokens=64,
                temperature=0,
                reasoning_effort="none",
            )
            print(f"  async response: {response.text.strip()!r}")
            test.check(async_marker.lower() in response.text.lower(), "async chat request returns the requested marker")

    asyncio.run(_run())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live ARC V2 inference client end-to-end checks.")
    parser.add_argument(
        "--url",
        default=os.environ.get("ARC_RUNTIME_URL", "http://127.0.0.1:7842"),
        help="Inference runtime server URL (default: ARC_RUNTIME_URL or http://127.0.0.1:7842)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("ARC_TEST_TIMEOUT_S", "300")),
        help="HTTP timeout in seconds (default: ARC_TEST_TIMEOUT_S or 300)",
    )
    parser.add_argument("--skip-agent", action="store_true", help="Skip the real model tool-call/agent test")
    parser.add_argument("--embeddings", action="store_true", help="Test embeddings (requires an embedding model configured on the runtime)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")

    test = SmokeTest()
    run_sync_checks(
        test,
        base_url=args.url,
        timeout=args.timeout,
        skip_agent=args.skip_agent,
        test_embeddings=args.embeddings,
    )
    run_async_checks(test, base_url=args.url, timeout=args.timeout)
    print(f"\nAll live end-to-end checks passed: {test.passed} assertions.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nEND-TO-END TEST FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
