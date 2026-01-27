from typing import List, Dict

from .base import Generator


class NoOpGenerator(Generator):
    """No-op generator that returns empty string."""
    
    def __init__(self):
        """Initialize no-op generator."""
        super().__init__()
    
    def generate(
        self,
        query: str,
        context: List[Dict],
    ) -> str:
        """
        Return empty string (no generation).
        
        Args:
            query: Query text (unused)
            context: List of context documents (unused)
        
        Returns:
            Empty string
        """
        return ""
