"""Chunking strategies for LinearRAG.

Provides three chunking strategies for text preprocessing before indexing:
1. OriginalChunking - Pass-through (no rechunking, may be truncated by embedding model)
2. FixedSizeChunking - Token-based chunking with overlap
3. SentenceBasedChunking - Semantic sentence grouping with overlap

Example:
    >>> from llmware.RAGMethods.LinearRAG.chunking import (
    ...     get_chunking_strategy, ChunkingConfig
    ... )
    >>> 
    >>> config = ChunkingConfig(fixed_max_tokens=300, fixed_overlap_tokens=50)
    >>> strategy = get_chunking_strategy("fixed_size", config)
    >>> chunks, metadata = strategy.chunk(raw_texts)
"""

from .config import ChunkingConfig
from .chunker import (
    Tokenizer,
    ChunkingStrategy,
    OriginalChunking,
    FixedSizeChunking,
    SentenceBasedChunking,
    get_chunking_strategy,
)

__all__ = [
    "ChunkingConfig",
    "Tokenizer",
    "ChunkingStrategy",
    "OriginalChunking",
    "FixedSizeChunking",
    "SentenceBasedChunking",
    "get_chunking_strategy",
]

