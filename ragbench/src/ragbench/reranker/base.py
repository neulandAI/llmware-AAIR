from abc import ABC, abstractmethod
from typing import List, Dict


class Reranker(ABC):
    """Base class for rerankers."""
    
    def __init__(self):
        """Initialize reranker (called once per run)."""
        pass
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        retrieved: List[Dict],
    ) -> List[Dict]:
        """
        Rerank retrieved documents.
        
        Args:
            query: Query text
            retrieved: List of retrieved document dicts with "text" and "doc_id"
        
        Returns:
            List of reranked document dicts with "doc_id", "score", "text"
        """
        pass
