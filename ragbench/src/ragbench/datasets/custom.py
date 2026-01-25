from pathlib import Path
from typing import Dict, List
import json

from .base import Dataset


class CustomDataset(Dataset):
    """Custom dataset loader from JSON file."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize custom dataset.
        
        Args:
            dataset_path: Path to JSON file with dataset
                Expected format:
                {
                    "queries": [{"id": "q1", "text": "query text"}, ...],
                    "corpus": {"doc1": {"text": "doc text", "title": "optional"}, ...},
                    "ground_truth": {"q1": ["doc1", "doc2"], ...}
                }
        """
        self.dataset_path = Path(dataset_path)
        self._data = None
    
    def load(self) -> Dict:
        """Load dataset from JSON file."""
        if self._data is None:
            with open(self.dataset_path, "r") as f:
                self._data = json.load(f)
        
        # Validate structure
        required_keys = ["queries", "corpus", "ground_truth"]
        for key in required_keys:
            if key not in self._data:
                raise ValueError(f"Dataset missing required key: {key}")
        
        # Convert ground_truth lists to sets
        ground_truth = {}
        for query_id, doc_ids in self._data["ground_truth"].items():
            ground_truth[query_id] = set(doc_ids) if isinstance(doc_ids, list) else doc_ids
        
        return {
            "queries": self._data["queries"],
            "corpus": self._data["corpus"],
            "ground_truth": ground_truth,
        }
