"""Client SDK for the ARC V2 inference-runtime service."""

from .agent import Agent, AsyncAgent
from .client import AsyncInferenceClient, InferenceClient
from .errors import AgentLoopError, InferenceAPIError, InferenceClientError, ToolExecutionError
from .tools import Tool, tool
from .types import (
    AgentEvent,
    AgentResult,
    AgentTrace,
    ChatContext,
    ChatMessage,
    ChatResponse,
    ConversationInfo,
    EmbeddingResponse,
    ModelInfo,
    StreamEvent,
    TokenizeResponse,
    TokenizedInput,
    ToolCall,
)

Client = InferenceClient
AsyncClient = AsyncInferenceClient
__version__ = "2.0.0"

__all__ = [
    "Agent",
    "AgentEvent",
    "AgentLoopError",
    "AgentResult",
    "AgentTrace",
    "AsyncAgent",
    "AsyncClient",
    "AsyncInferenceClient",
    "ChatContext",
    "ChatMessage",
    "ChatResponse",
    "Client",
    "ConversationInfo",
    "EmbeddingResponse",
    "InferenceAPIError",
    "InferenceClient",
    "InferenceClientError",
    "ModelInfo",
    "StreamEvent",
    "TokenizeResponse",
    "TokenizedInput",
    "Tool",
    "ToolCall",
    "ToolExecutionError",
    "tool",
]
