<div align="center">

# ARC V2 Inference Client

**A small Python client for the ARC V2 inference runtime.**

</div>

The client wraps the runtime's HTTP API and keeps application logic on the client side. It supports direct chat/completions, streaming, model metadata, exact token/context inspection, persistent conversations, embeddings, and a basic tool-agent loop.

## Install

```bash
uv add git+https://github.com/ARC-V2-AI/inference-client.git
```

## Quick start

```python
from arc_v2_inference_client import InferenceClient

with InferenceClient("http://127.0.0.1:7842") as client:
    response = client.chat.create([
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain ARC V2 in one sentence."},
    ], temperature=0.2)
    print(response.text)
```

Pass either the server root (`http://127.0.0.1:7842`) or the API root (`http://127.0.0.1:7842/v1`). The client normalizes both forms.

## Runtime-native APIs

### Model and tokenization

```python
model = client.model.get()
print(model.id, model.tool_protocol, model.supports_tools)

counts = client.tokenize(
    ["First prompt", "Second prompt"],
    include_tokens=False,
)
print(counts.token_counts, counts.total_tokens)
```

`client.models.list()` calls `/v1/models`; `client.model.get()` calls the runtime's `/v1/model`. Tokenization uses `/v1/tokenize`, which can optionally return token IDs.

### Exact context accounting

```python
context = client.chat.context(
    [{"role": "user", "content": "Can this prompt fit?"}],
    max_output_tokens=512,
    tools=[],
)

print(context.prompt_tokens, context.fits, context.remaining_tokens)
```

Context accounting calls `/v1/chat/context` so the runtime can render and count the prompt using the loaded model's actual chat template. This is more reliable than estimating tokens from message character length.

### Persisted conversations

```python
with InferenceClient() as client:
    conversation = client.conversation(
        system="You are a helpful, concise assistant.",
        title="First conversation",
    )

    print(conversation.send("Hello").text)
    print(conversation.send("What did I just say?").text)
    print(conversation.id)

    saved = client.conversations.get(conversation.id)
    print(saved.messages)
```

Conversations are stored by the runtime. The client sends only new messages to `/v1/conversations/{id}/chat/completions`; it does **not** resend the transcript on every turn. Use `client.conversations.list()` and `client.conversations.delete(id)` for transcript management. Context pruning during generation does not erase the stored transcript.

### Streaming

```python
for event in conversation.stream("Explain tool calling"):
    if event.type == "text":
        print(event.text, end="", flush=True)
```

The stream exposes text, reasoning fragments, tool-call deltas, finish information, and the raw event payload. The caller controls what to display or record.

### Embeddings

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

The runtime may return HTTP 503 when its dedicated embedding model is not configured. The client surfaces the server error rather than silently switching to the chat model.

## Tools and agents

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
    )
    result = agent.run("What is 12 + 30?", trace=True)
    print(result.text)
    print(result.conversation_id)
    for event in result.trace.events:
        print(event.type, event.data)
```

The agent executes registered Python functions locally. Tool schemas are inferred from type hints and docstrings; provide an explicit JSON Schema to `@tool(parameters=...)` when inference is insufficient. Only register tools that are appropriate for the prompts and permissions in your application.

For async applications, use `AsyncInferenceClient` and `await client.chat.create(...)`, `await client.conversation(...)`, or `await client.agent(...).run(...)`. Async agents support both normal and `async def` tools.

## Stateless API surface

| Method | Runtime route | Purpose |
| --- | --- | --- |
| `client.chat.create(...)` | `POST /v1/chat/completions` | One stateless chat request |
| `client.completions.create(...)` | `POST /v1/completions` | Text completion |
| `client.embeddings.create(...)` | `POST /v1/embeddings` | Vector embeddings |
| `client.model.get()` | `GET /v1/model` | Loaded-model metadata and capabilities |
| `client.models.list()` | `GET /v1/models` | Model listing |
| `client.tokenize(...)` | `POST /v1/tokenize` | Native tokenizer counts / IDs |
| `client.chat.context(...)` | `POST /v1/chat/context` | Exact rendered-prompt budget |
| `client.conversations.*` | `/v1/conversations` | Persistent transcript lifecycle |

## Migration and naming

The distribution and import names are now aligned with the server counterpart:

- Distribution: `arc-v2-inference-client`
- Import: `arc_v2_inference_client`
- Server distribution: `arc-v2-inference-runtime`
- Server import: `arc_v2_inference_runtime`

The existing `arc_inference` package can remain in the project for a migration period. New application code should use `arc_v2_inference_client` so the client/runtime relationship is explicit. Rename the GitHub repository to `ARC-V2-AI/inference-client` when repository ownership and downstream install URLs are ready to be updated.

## Design boundaries

- The runtime owns model loading, tokenization, actual prompt rendering, context-window enforcement, and persistent transcript storage.
- The client owns HTTP transport, typed response helpers, local tool execution, and application-facing convenience methods.
- Conversation generation sends only newly added messages. Sending the full stored transcript again duplicates history.
- The SDK does not infer server-side model capabilities from model names; use `/v1/model` metadata.
