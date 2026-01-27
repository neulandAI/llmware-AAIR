from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Configuration for RAG pipeline experiments."""
    
    # Global settings
    top_k: int = 10
    results_path: Optional[str] = None
    
    # Parameter values for sweeps (component constructors are in main.py)
    chunk_sizes: List[int] = field(default_factory=lambda: [500, 1000])
    overlaps: List[int] = field(default_factory=lambda: [50, 100])
    dataset_paths: List[str] = field(default_factory=lambda: ["data/dataset.json"])
    
    def __post_init__(self):
        """Set default results_path if not provided."""
        if self.results_path is None:
            self.results_path = "results/experiment_results.jsonl"
