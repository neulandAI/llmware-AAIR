"""Configuration for chunking strategies."""

from dataclasses import dataclass


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies.
    
    Attributes:
        fixed_max_tokens: Maximum tokens per chunk for fixed-size chunking (default: 300).
            Should be under the embedding model's max sequence length with margin.
        fixed_overlap_tokens: Token overlap between chunks for fixed-size chunking (default: 50).
        
        sentences_per_chunk: Number of sentences to group per chunk for sentence-based
            chunking (default: 4).
        sentence_overlap: Number of overlapping sentences between chunks (default: 1).
        max_chunk_tokens: Maximum tokens for sentence-based chunks (default: 350).
        
        min_chunk_tokens: Minimum tokens to keep a chunk; smaller chunks are filtered
            out (default: 15).
        
        embedding_model_name: Model name for tokenization. Should match the embedding
            model used for accurate token counting (default: sentence-transformers/all-mpnet-base-v2).
    """
    
    # Fixed-size chunking parameters (in tokens)
    fixed_max_tokens: int = 300
    fixed_overlap_tokens: int = 50
    
    # Sentence-based chunking parameters
    sentences_per_chunk: int = 4
    sentence_overlap: int = 1
    max_chunk_tokens: int = 350
    
    # Common parameters
    min_chunk_tokens: int = 15
    
    # Model for tokenization
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"

