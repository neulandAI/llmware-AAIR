"""LinearRAG - Graph-based Retrieval-Augmented Generation.

A relation-free graph construction method for efficient GraphRAG that uses
Named Entity Recognition and Personalized PageRank for multi-hop reasoning.

Features:
- Context-preserving entity extraction via SpaCy NER
- Graph-based retrieval with Personalized PageRank
- Dense passage retrieval fallback
- Flexible embedding model support (SentenceTransformer, llmware)
- llmware LLM integration for QA
- Evaluation support via LinearRAGQueryAdapter

Example:
    >>> from llmware.models import ModelCatalog
    >>> from llmware.RAGMethods.LinearRAG import LinearRAG, LinearRAGConfig
    >>> from llmware.RAGMethods.LinearRAG.adapters import (
    ...     SentenceTransformerAdapter, LLMWareModelWrapper
    ... )
    >>> 
    >>> # Setup
    >>> llm = ModelCatalog().load_model("llmware/bling-phi-3-gguf")
    >>> config = LinearRAGConfig(
    ...     dataset_name="my_data",
    ...     embedding_model=SentenceTransformerAdapter("all-mpnet-base-v2"),
    ...     llm_model=LLMWareModelWrapper(llm),
    ...     working_dir="./linearrag_data"
    ... )
    >>> 
    >>> # Run
    >>> rag = LinearRAG(config)
    >>> rag.index(passages)
    >>> results = rag.qa(questions)

Evaluation Example:
    >>> from llmware.RAGMethods.LinearRAG import LinearRAG, LinearRAGConfig, LinearRAGQueryAdapter
    >>> from llmware.evaluation import RetrievalEvaluator
    >>> 
    >>> # Index with metadata for evaluation
    >>> passages = [{"text": "...", "file_source": "doc.pdf", "page_num": 1}, ...]
    >>> rag.index(passages)
    >>> 
    >>> # Evaluate using adapter
    >>> adapter = LinearRAGQueryAdapter(rag)
    >>> evaluator = RetrievalEvaluator.__new__(RetrievalEvaluator)
    >>> evaluator.query = adapter
    >>> evaluator.dataset_path = "qa_dataset.json"
    >>> evaluator.metrics = ["hit_rate", "mrr", "recall", "ndcg"]
    >>> results = evaluator.evaluate(top_k=10)
"""

from .core.linear_rag import LinearRAG
from .core.config import LinearRAGConfig
from .core.graph_builder import GraphBuilder

from .adapters.embedding_adapter import (
    EmbeddingModelProtocol,
    SentenceTransformerAdapter,
    LLMWareEmbeddingAdapter,
)
from .adapters.llm_wrapper import (
    LLMProtocol,
    LLMWareModelWrapper,
    OpenAILLMAdapter,
    HuggingFaceLLMAdapter,
)
from .adapters.evaluation_adapter import LinearRAGQueryAdapter

from .ner.spacy_ner import SpacyNER

from .storage.embedding_store import EmbeddingStore

from .chunking import (
    ChunkingConfig,
    Tokenizer,
    ChunkingStrategy,
    OriginalChunking,
    FixedSizeChunking,
    SentenceBasedChunking,
    get_chunking_strategy,
)

from .utils import (
    compute_mdhash_id,
    normalize_answer,
    min_max_normalize,
    strip_passage_prefix,
)

__all__ = [
    "LinearRAG",
    "LinearRAGConfig",
    "GraphBuilder",
    "EmbeddingModelProtocol",
    "SentenceTransformerAdapter",
    "LLMWareEmbeddingAdapter",
    "LLMProtocol",
    "LLMWareModelWrapper",
    "OpenAILLMAdapter",
    "HuggingFaceLLMAdapter",
    "LinearRAGQueryAdapter",
    "SpacyNER",
    "EmbeddingStore",
    "ChunkingConfig",
    "Tokenizer",
    "ChunkingStrategy",
    "OriginalChunking",
    "FixedSizeChunking",
    "SentenceBasedChunking",
    "get_chunking_strategy",
    "compute_mdhash_id",
    "normalize_answer",
    "min_max_normalize",
    "strip_passage_prefix",
]
