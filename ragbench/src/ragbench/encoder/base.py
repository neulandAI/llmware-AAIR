from abc import ABC, abstractmethod
from typing import List


class Encoder(ABC):
    """Base class for embedding encoders."""
    
    def __init__(self):
        """Initialize encoder (called once per run)."""
        pass
    
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of text strings to encode
        
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name/identifier for this encoder."""
        pass
