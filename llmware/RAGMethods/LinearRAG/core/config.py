"""Configuration for LinearRAG."""

from dataclasses import dataclass
from typing import Any


@dataclass
class LinearRAGConfig:
    """Configuration for LinearRAG.
    
    Attributes:
        dataset_name: Name of the dataset (used for storage paths)
        embedding_model: Embedding model implementing EmbeddingModelProtocol
        llm_model: LLM model implementing LLMProtocol (for QA)
        spacy_model: SpaCy model name for NER (default: en_core_web_trf)
        working_dir: Base directory for storing embeddings and graph
        batch_size: Batch size for embedding computation
        max_workers: Maximum workers for parallel processing
        retrieval_top_k: Number of top passages to retrieve
        max_iterations: Maximum iterations for entity graph traversal
        top_k_sentence: Top K sentences to consider per entity
        passage_ratio: Weight ratio for passage scores in PPR
        passage_node_weight: Weight for passage nodes in graph
        damping: Damping factor for Personalized PageRank
        iteration_threshold: Minimum score threshold for entity iteration
        chunk_token_size: Token size for chunking (if chunking is needed)
        chunk_overlap_token_size: Overlap size for chunking
    """
    
    # Required
    dataset_name: str
    
    # Models - use Any to avoid circular imports, actual types are protocol-based
    embedding_model: Any = None
    llm_model: Any = None
    
    # SpaCy configuration
    spacy_model: str = "en_core_web_trf"
    
    # Storage
    working_dir: str = "./linearrag_data"
    
    # Processing
    batch_size: int = 128
    max_workers: int = 16
    
    # Retrieval parameters
    retrieval_top_k: int = 3
    max_iterations: int = 3
    top_k_sentence: int = 1
    
    # Scoring parameters
    passage_ratio: float = 1.5
    passage_node_weight: float = 0.05
    damping: float = 0.5
    iteration_threshold: float = 0.5
    
    # Chunking
    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.dataset_name:
            raise ValueError("dataset_name is required")
        
        if self.retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be at least 1")
        
        if not 0 < self.damping < 1:
            raise ValueError("damping must be between 0 and 1")
        
        if self.iteration_threshold < 0:
            raise ValueError("iteration_threshold must be non-negative")

