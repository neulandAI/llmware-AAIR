from abc import ABC, abstractmethod
from typing import List, Dict


class Chunker(ABC):
    """Base class for document chunkers."""
    
    def __init__(self):
        """Initialize chunker (called once per run)."""
        pass
    
    @abstractmethod
    def chunk(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk documents into smaller pieces.
        
        Args:
            documents: List of document dicts with at least "text" key
        
        Returns:
            List of chunk dicts with "text" and optionally "doc_id", "chunk_id", "metadata"
        """
        pass
