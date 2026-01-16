"""Adapters for embedding models, LLM wrappers, and evaluation compatibility."""

from .embedding_adapter import (
    EmbeddingModelProtocol,
    SentenceTransformerAdapter,
    LLMWareEmbeddingAdapter,
)
from .llm_wrapper import (
    LLMProtocol,
    LLMWareModelWrapper,
    OpenAILLMAdapter,
    HuggingFaceLLMAdapter,
)
from .evaluation_adapter import LinearRAGQueryAdapter

__all__ = [
    "EmbeddingModelProtocol",
    "SentenceTransformerAdapter",
    "LLMWareEmbeddingAdapter",
    "LLMProtocol",
    "LLMWareModelWrapper",
    "OpenAILLMAdapter",
    "HuggingFaceLLMAdapter",
    "LinearRAGQueryAdapter",
]
