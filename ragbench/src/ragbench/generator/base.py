from abc import ABC, abstractmethod
from typing import List, Dict


class Generator(ABC):
    """Base class for generators."""
    
    def __init__(self):
        """Initialize generator (called once per run)."""
        pass
    
    @abstractmethod
    def generate(
        self,
        query: str,
        context: List[Dict],
    ) -> str:
        """
        Generate response given query and context.
        
        Args:
            query: Query text
            context: List of context document dicts with "text"
        
        Returns:
            Generated response text
        """
        pass
