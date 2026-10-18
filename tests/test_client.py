from __future__ import annotations

import json

import httpx
import pytest

from arc_v2_inference_client import InferenceClient, tool


def response(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/v1/model":
        return httpx.Response(200, json={"id": "local-model", "supports_tools": True, "tool_protocol": "qwen_xml"})
    if path == "/v1/models":
        return httpx.Response(200, json={"data": [{"id": "local-model"}]})
    if path == "/v1/tokenize":
        body = json.loads(request.content)
        inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
        return httpx.Response(200, json={"data": [{"token_count": len(value.split()), "tokens": [1, 2] if body["include_tokens"] else []} for value in inputs]})
    if path == "/v1/chat/context":
        return httpx.Response(200, json={"prompt_tokens": 9, "fits": True, "context_window": 4096, "remaining_tokens": 4000})
    if path == "/v1/chat/completions":
        return httpx.Response(200, json={"id": "chat-1", "model": "local-model", "choices": [{"message": {"role": "assistant", "content": "Hello back"}, "finish_reason": "stop"}]})
    if path == "/v1/conversations":
        if request.method == "POST":
            body = json.loads(request.content)
            return httpx.Response(200, json={"id": "conv-1", "title": body.get("title"), "messages": []})
        return httpx.Response(200, json={"data": [{"id": "conv-1", "title": "saved"}]})
    if path == "/v1/conversations/conv-1/chat/completions":
        body = json.loads(request.content)
        # The system message is sent once with the first turn; later calls only
        # contain the new user/tool messages, not a replayed transcript.
        messages = body["messages"]
        assert messages[-1]["role"] == "user"
        assert len(messages) in (1, 2)
        if len(messages) == 2:
            assert messages[0] == {"role": "system", "content": "Be concise"}
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Persisted reply"}, "finish_reason": "stop"}]})
    if path == "/v1/embeddings":
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2], "index": 0}], "usage": {"prompt_tokens": 2}})
    if path == "/v1/conversations/conv-1":
        return httpx.Response(200, json={"id": "conv-1", "title": "saved", "messages": [{"role": "user", "content": "hi"}]})
    return httpx.Response(404, json={"detail": f"Unhandled path: {path}"})


def test_model_tokenizer_and_context() -> None:
    client = InferenceClient("http://runtime.local", transport=httpx.MockTransport(response))
    try:
        model = client.model.get()
        assert model.id == "local-model"
        assert model.supports_tools is True
        assert client.models.list()["data"][0]["id"] == "local-model"

        tokens = client.tokenize(["one two", "three"], include_tokens=True)
        assert tokens.token_counts == [2, 1]
        assert tokens.total_tokens == 3
        assert tokens.data[0].tokens == (1, 2)

        context = client.chat.context([{"role": "user", "content": "Can I fit?"}], max_output_tokens=128)
        assert context.prompt_tokens == 9
        assert context.fits is True
    finally:
        client.close()


def test_stateless_chat_conversation_and_embeddings() -> None:
    client = InferenceClient("http://runtime.local/v1", transport=httpx.MockTransport(response))
    try:
        answer = client.chat.create([{"role": "user", "content": "Hello"}], temperature=0.2)
        assert answer.text == "Hello back"

        conversation = client.conversation(system="Be concise", title="test")
        answer = conversation.send("Hello")
        assert answer.text == "Persisted reply"
        answer = conversation.send("Continue")
        assert answer.text == "Persisted reply"
        saved = client.conversations.get(conversation.id)
        assert saved.id == "conv-1"
        assert saved.messages[0]["role"] == "user"

        embedding = client.embeddings.create(["a passage"], input_type="document")
        assert embedding.data[0]["embedding"] == [0.1, 0.2]
    finally:
        client.close()


def test_streaming_sse_decodes_text_and_tool_deltas() -> None:
    frames = [
        'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"add","arguments":"{\\"a\\":"}}]},"finish_reason":null}]}\n\n',
        'data: [DONE]\n\n',
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, text="".join(frames), headers={"content-type": "text/event-stream"})

    client = InferenceClient("http://runtime.local", transport=httpx.MockTransport(handler))
    try:
        events = list(client.chat.create([{"role": "user", "content": "go"}], stream=True))
        assert events[0].text == "Hi"
        assert events[1].tool_calls[0].name == "add"
        assert events[1].tool_calls[0].arguments == '{"a":'
    finally:
        client.close()


def test_tool_schema_and_local_agent_loop() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    spec = getattr(add, "__arc_v2_tool__")
    definition = spec.definition()
    assert definition["function"]["parameters"]["required"] == ["a", "b"]
    assert spec.execute('{"a": 2, "b": 3}') == "5"


@pytest.mark.asyncio
async def test_async_model_and_tokenizer() -> None:
    client = httpx.MockTransport(response)
    async with __import__("arc_v2_inference_client").AsyncInferenceClient(
        "http://runtime.local", transport=client
    ) as async_client:
        model = await async_client.model.get()
        assert model.id == "local-model"
        tokenized = await async_client.tokenize("one two")
        assert tokenized.token_counts == [2]


def _agent_runtime_response(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/v1/conversations" and request.method == "POST":
        return httpx.Response(200, json={"id": "conv-agent", "title": "agent test", "messages": []})
    if path == "/v1/conversations/conv-agent/chat/completions":
        body = json.loads(request.content)
        messages = body["messages"]
        assert messages
        if any(message.get("role") == "tool" for message in messages):
            assert len(messages) == 1  # Runtime already retained the assistant tool call.
            assert messages[0]["content"] in {"5", "6"}
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "The result is ready."}, "finish_reason": "stop"}]})
        assert messages[-1]["role"] == "user"
        return httpx.Response(200, json={"choices": [{"message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-1", "type": "function", "function": {
                "name": "calculate", "arguments": '{"a": 2, "b": 3}'
            }}],
        }, "finish_reason": "tool_calls"}]})
    return httpx.Response(404, json={"detail": f"Unhandled agent path: {path}"})


def test_agent_executes_tools_and_returns_trace() -> None:
    @tool(name="calculate", description="Add two integers.")
    def add(a: int, b: int) -> int:
        return a + b

    client = InferenceClient("http://runtime.local", transport=httpx.MockTransport(_agent_runtime_response))
    callback_events = []
    try:
        agent = client.agent(instructions="Use tools when useful.", tools=[add], trace_callback=callback_events.append)
        result = agent.run("Calculate 2 + 3", trace=True)
        assert result.text == "The result is ready."
        assert result.conversation_id == "conv-agent"
        assert result.trace is not None
        assert [item["name"] for item in result.trace.tool_summary()] == ["calculate"]
        assert result.trace.tool_summary()[0]["result"] == "5"
        assert any(event.type == "tool_call_completed" for event in callback_events)
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_agent_executes_async_tools() -> None:
    from arc_v2_inference_client import AsyncInferenceClient

    @tool(name="calculate")
    async def multiply(a: int, b: int) -> int:
        return a * b

    async with AsyncInferenceClient(
        "http://runtime.local", transport=httpx.MockTransport(_agent_runtime_response)
    ) as client:
        agent = client.agent(tools=[multiply])
        result = await agent.run("Calculate 2 * 3")
        assert result.text == "The result is ready."
        assert result.conversation_id == "conv-agent"


def test_agent_stream_saves_result_and_trace() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/conversations" and request.method == "POST":
            return httpx.Response(200, json={"id": "conv-stream", "title": "stream test", "messages": []})
        if request.url.path == "/v1/conversations/conv-stream/chat/completions":
            frames = (
                'data: {"id":"stream-1","model":"local-model","choices":[{"index":0,"delta":{"content":"Streamed "},"finish_reason":null}]}\n\n'
                'data: {"id":"stream-1","model":"local-model","choices":[{"index":0,"delta":{"content":"reply."},"finish_reason":null}]}\n\n'
                'data: {"id":"stream-1","model":"local-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                'data: [DONE]\n\n'
            )
            return httpx.Response(200, text=frames, headers={"content-type": "text/event-stream"})
        return httpx.Response(404, json={"detail": "not found"})

    client = InferenceClient("http://runtime.local", transport=httpx.MockTransport(handler))
    try:
        agent = client.agent()
        yielded = list(agent.stream("Say hello", trace=True))
        assert any(getattr(item, "type", None) == "text" for item in yielded)
        assert agent.last_result is not None
        assert agent.last_result.text == "Streamed reply."
        assert agent.last_result.trace is not None
        assert agent.last_result.trace.duration_ms >= 0
    finally:
        client.close()
