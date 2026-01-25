from typing import List

from .base import Encoder


class Voyage3Large(Encoder):
    """Voyage 3 Large embedding model."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Voyage 3 Large encoder.
        
        Args:
            api_key: API key for Voyage (if None, tries to get from environment)
        """
        super().__init__()
        # Try to use llmware interface first, fallback to direct API
        try:
            from llmware.models import ModelCatalog
            self.model = ModelCatalog().load_model("voyage-large-3")
            self.use_llmware = True
        except Exception:
            # Fallback to direct Voyage API
            try:
                import voyageai
                self.client = voyageai.Client(api_key=api_key)
                self.use_llmware = False
            except ImportError:
                raise ImportError("voyageai package required for Voyage3Large encoder")
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts using Voyage 3 Large."""
        if self.use_llmware:
            embeddings = []
            for text in texts:
                embedding = self.model.embedding(text)
                embeddings.append(embedding)
            return embeddings
        else:
            # Use Voyage API directly
            result = self.client.embed(texts, model="voyage-large-3")
            return result.embeddings
    
    def get_model_name(self) -> str:
        return "voyage-large-3"
