from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ChunkerConfig:
    """Configuration for chunkers."""
    chunk_sizes: List[int] = field(default_factory=lambda: [500, 1000])
    overlaps: List[int] = field(default_factory=lambda: [50, 100])


@dataclass
class ParserConfig:
    """Configuration for parsers."""
    parser_types: List[str] = field(default_factory=lambda: ["unstructured", "llmware"])


@dataclass
class EncoderConfig:
    """Configuration for encoders."""
    encoder_types: List[str] = field(default_factory=lambda: ["qwen_4b", "qwen_8b"])


@dataclass
class RetrieverConfig:
    """Configuration for retrievers."""
    retriever_types: List[str] = field(default_factory=lambda: ["semantic", "text", "hybrid"])


@dataclass
class DatasetConfig:
    """Configuration for datasets."""
    dataset_paths: List[str] = field(default_factory=lambda: ["data/dataset.json"])


@dataclass
class Config:
    """Configuration for RAG pipeline experiments."""
    
    # Global settings
    top_k: int = 10
    results_path: Optional[str] = None
    
    # Component configurations
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    def __post_init__(self):
        """Set default results_path if not provided."""
        if self.results_path is None:
            self.results_path = "results/experiment_results.jsonl"
