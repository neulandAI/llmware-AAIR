from typing import List, Dict

from .base import Reranker


class NoOpReranker(Reranker):
    """No-op reranker that returns documents unchanged."""
    
    def __init__(self):
        """Initialize no-op reranker."""
        super().__init__()
    
    def rerank(
        self,
        query: str,
        retrieved: List[Dict],
    ) -> List[Dict]:
        """
        Return retrieved documents unchanged (no reranking).
        
        Args:
            query: Query text (unused)
            retrieved: List of retrieved document dicts
        
        Returns:
            Same list of documents unchanged
        """
        return retrieved
