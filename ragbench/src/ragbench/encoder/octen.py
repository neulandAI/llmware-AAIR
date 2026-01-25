from typing import List

from .base import Encoder


class OctenEmbedding4B(Encoder):
    """Octen Embedding 4B model."""
    
    def __init__(self):
        super().__init__()
        try:
            from llmware.models import ModelCatalog
            self.model = ModelCatalog().load_model("octen-embedding-4b")
        except Exception as e:
            raise ImportError(f"Failed to load Octen Embedding 4B: {e}")
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts using Octen Embedding 4B."""
        embeddings = []
        for text in texts:
            embedding = self.model.embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def get_model_name(self) -> str:
        return "octen-embedding-4b"


class OctenEmbedding8B(Encoder):
    """Octen Embedding 8B model."""
    
    def __init__(self):
        super().__init__()
        try:
            from llmware.models import ModelCatalog
            self.model = ModelCatalog().load_model("octen-embedding-8b")
        except Exception as e:
            raise ImportError(f"Failed to load Octen Embedding 8B: {e}")
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts using Octen Embedding 8B."""
        embeddings = []
        for text in texts:
            embedding = self.model.embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def get_model_name(self) -> str:
        return "octen-embedding-8b"
