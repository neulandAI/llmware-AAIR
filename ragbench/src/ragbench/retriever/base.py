from abc import ABC, abstractmethod
from typing import List, Dict


class Retriever(ABC):
    """Base class for retrievers."""
    
    def __init__(self):
        """Initialize retriever (called once per run)."""
        pass
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        library,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Retrieve documents for a query.
        
        Args:
            query: Query text
            library: llmware Library instance with embedded documents
            top_k: Number of results to return
        
        Returns:
            List of result dicts with at least "doc_id" and "score" keys
        """
        pass
