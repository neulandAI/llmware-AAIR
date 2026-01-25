from typing import List, Dict

from .base import Reranker


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model
        """
        super().__init__()
        self.model_name = model_name
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except ImportError:
            raise ImportError("sentence-transformers required for CrossEncoderReranker")
    
    def rerank(
        self,
        query: str,
        retrieved: List[Dict],
    ) -> List[Dict]:
        """Rerank retrieved documents using cross-encoder."""
        if not retrieved:
            return []
        
        # Prepare pairs for scoring
        pairs = [(query, doc.get("text", "")) for doc in retrieved]
        
        # Score pairs
        scores = self.model.predict(pairs)
        
        # Combine scores with documents and sort
        reranked = []
        for doc, score in zip(retrieved, scores):
            reranked.append({
                "doc_id": doc.get("doc_id", ""),
                "score": float(score),
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
            })
        
        # Sort by score (descending)
        reranked.sort(key=lambda x: x["score"], reverse=True)
        
        return reranked
