<div align="center">

# ARC V2 Inference Client

**Python client and local agent SDK for the ARC V2 inference runtime.**

</div>

The client provides stateless chat/completion requests, streaming, token/context inspection, runtime resource controls, embeddings and synchronous/asynchronous tool agents. Agent message history is owned by the client process; the inference server does not persist conversations.

## Install

```bash
uv add git+https://github.com/ARC-V2-AI/inference-client.git
```

## Quick start

```python
from arc_v2_inference_client import InferenceClient

with InferenceClient("http://127.0.0.1:7842") as client:
    response = client.chat.create(
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Explain ARC V2 in one sentence."},
        ],
        stream=False,
        max_output_tokens=256,
        context_limit=4096,
        context_overflow="reject",
    )
    print(response.text)
    print(response.usage)
```

Pass either the server root (`http://127.0.0.1:7842`) or `/v1` API root. The client normalizes both forms.

## Token and context control

```python
with InferenceClient() as client:
    tokens = client.tokenize("How many tokens is this?")
    print(tokens.total_tokens)

    context = client.chat.context(
        [{"role": "user", "content": "Can this prompt fit?"}],
        max_output_tokens=512,
        context_limit=4096,
        context_overflow="reject",
    )
    print(context.prompt_tokens, context.fits, context.remaining_tokens)
```

The server renders messages using the loaded model's native chat template and counts them with its tokenizer. `context_limit` can restrict one request below the server's configured window. `context_overflow` accepts `"reject"` or `"truncate"` (oldest complete user turns are removed only in truncate mode).

Responses expose token usage:

```python
print(response.usage)
# {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
```

## Local agent history

```python
from arc_v2_inference_client import InferenceClient, tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

with InferenceClient() as client:
    agent = client.agent(
        instructions="Use tools when helpful. Be concise.",
        tools=[add],
        max_iterations=8,
        max_output_tokens=512,
    )

    first = agent.run("What is 12 + 30?")
    second = agent.run("Multiply that result by two.")
    print(first.text, second.text)

    # History remains in this Agent instance, not on the server.
    state = agent.export_state()
    agent.reset()
    agent.restore_state(state)
```

The agent sends its complete local message list with each inference request. Assistant tool-call messages and tool results are appended locally so multi-step tool loops remain coherent. `reset()` discards that agent's working transcript. `export_state()` and `restore_state()` let the caller persist or restore it using its own storage mechanism.

Agents are independent client-side objects. A main agent can pause while a temporary subagent runs through the same inference client. The runtime serializes their inference requests and clears shared KV contents after each generation. The subagent should be discarded after its result is returned; its local transcript is then released with the object. This does not create per-agent GPU allocations or retained KV sessions.

## Streaming

```python
for event in client.chat.create(
    [{"role": "user", "content": "Explain tool calling."}],
    stream=True,
):
    if event.type == "text":
        print(event.text, end="", flush=True)
```

The stream exposes text, reasoning fragments, tool-call deltas, finish information and the raw event payload. Final usage is included in the final event's raw data.

## Embeddings

```python
embeddings = client.embeddings.create(
    ["Passage to index", "Another passage"],
    input_type="document",
)

query_embedding = client.embeddings.create(
    "Find passages about local inference",
    input_type="query",
    instruction="Retrieve documents relevant to the user's question.",
)
```

The runtime returns HTTP 503 when a dedicated embedding model is not configured and fallback is disabled.

## Runtime resource inspection

```python
with InferenceClient() as client:
    print(client.runtime.status())
    print(client.runtime.memory())
    client.runtime.clear_cache()
```

`clear_cache()` is a global administrative operation that clears the active shared KV sequence. The shared KV allocation remains reserved for reuse; it is not an operation that returns the model's allocated VRAM to the operating system. Do not call it concurrently with work that expects retained inference state. The current runtime has no retained per-agent KV sessions.

## Async usage

```python
from arc_v2_inference_client import AsyncInferenceClient

async with AsyncInferenceClient() as client:
    agent = client.agent(instructions="You are concise.")
    result = await agent.run("Hello")
    print(result.text)
    print(await client.runtime.memory())
```

Async agents support synchronous and `async def` tools.

## API surface

| Client method | Runtime route | Purpose |
|---|---|---|
| `client.chat.create(...)` | `POST /v1/chat/completions` | Stateless chat and streaming |
| `client.completions.create(...)` | `POST /v1/completions` | Text completion |
| `client.embeddings.create(...)` | `POST /v1/embeddings` | Embeddings |
| `client.model.get()` | `GET /v1/model` | Model metadata |
| `client.models.list()` | `GET /v1/models` | Model listing |
| `client.tokenize(...)` | `POST /v1/tokenize` | Token counts and IDs |
| `client.chat.context(...)` | `POST /v1/context/inspect` | Rendered prompt budget |
| `client.runtime.status()` | `GET /v1/runtime/status` | Runtime and queue state |
| `client.runtime.memory()` | `GET /v1/runtime/memory` | Process and GPU memory snapshot |
| `client.runtime.clear_cache()` | `POST /v1/runtime/cache/clear` | Global KV-content clear |

## Migration from persisted conversations

- Replace `client.conversation(...)` with `client.agent(...)` or maintain your own `messages` list and call `client.chat.create(messages, stream=False)`.
- Replace `client.conversations.*` with caller-owned persistence using `Agent.export_state()` / `restore_state()`.
- Each chat request must now include the complete message history.
- `AgentResult` no longer exposes a server conversation ID; agent history belongs to the client.

The previous SQLite conversation database is no longer read or modified by the runtime. Back it up or export its contents before removing it manually.

## Design boundaries

- Runtime: model lifecycle, native rendering/tokenization, context enforcement, serialized inference and KV cleanup.
- Client/Agent: working transcript, tool execution, agent orchestration and local state export/restore.
- ARC Memory Manager: long-term/episodic/semantic memory and summarization should remain a separate ARC service or client-side component.
- KV cache: an inference performance resource, not persistent agent memory.

The runtime API currently has no authentication. Keep the service bound to localhost unless protected by an authenticated gateway.
