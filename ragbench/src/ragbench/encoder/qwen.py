from typing import List

from .base import Encoder


class QwenEmbedding4B(Encoder):
    """Qwen Embedding 4B model using llmware interface."""
    
    def __init__(self):
        super().__init__()
        try:
            from llmware.models import ModelCatalog
            # Initialize llmware embedding model
            self.model = ModelCatalog().load_model("qwen-embedding-4b")
        except Exception as e:
            raise ImportError(f"Failed to load Qwen Embedding 4B: {e}")
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts using Qwen Embedding 4B."""
        embeddings = []
        for text in texts:
            # Use llmware embedding interface
            embedding = self.model.embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def get_model_name(self) -> str:
        return "qwen-embedding-4b"


class QwenEmbedding8B(Encoder):
    """Qwen Embedding 8B model using llmware interface."""
    
    def __init__(self):
        super().__init__()
        try:
            from llmware.models import ModelCatalog
            self.model = ModelCatalog().load_model("qwen-embedding-8b")
        except Exception as e:
            raise ImportError(f"Failed to load Qwen Embedding 8B: {e}")
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts using Qwen Embedding 8B."""
        embeddings = []
        for text in texts:
            embedding = self.model.embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def get_model_name(self) -> str:
        return "qwen-embedding-8b"
