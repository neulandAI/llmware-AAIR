from abc import ABC, abstractmethod
from typing import Dict, List


class Dataset(ABC):
    """Base class for dataset loaders."""
    
    @abstractmethod
    def load(self) -> Dict:
        """
        Load the dataset.
        
        Returns:
            Dictionary with keys:
                - "queries": List of query dicts with "id" and "text"
                - "corpus": Dict mapping doc_id to doc dict with "text" and optionally "title"
                - "ground_truth": Dict mapping query_id to set of relevant doc_ids
        """
        pass
    
    def get_queries(self) -> List[Dict]:
        """Get list of queries."""
        data = self.load()
        return data.get("queries", [])
    
    def get_corpus(self) -> Dict[str, Dict]:
        """Get document corpus."""
        data = self.load()
        return data.get("corpus", {})
    
    def get_ground_truth(self) -> Dict[str, set]:
        """Get ground truth relevance judgments."""
        data = self.load()
        return data.get("ground_truth", {})
