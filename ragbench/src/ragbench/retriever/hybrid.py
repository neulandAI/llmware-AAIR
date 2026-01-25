from typing import List, Dict

from .base import Retriever


class HybridRetriever(Retriever):
    """Hybrid retriever using llmware Query.hybrid_search."""
    
    def __init__(self, embedding_distance_threshold: float = 1.0, semantic_weight: float = 0.5):
        """
        Initialize hybrid retriever.
        
        Args:
            embedding_distance_threshold: Maximum embedding distance threshold
            semantic_weight: Weight for semantic search (0.0-1.0), text weight is 1.0 - semantic_weight
        """
        super().__init__()
        self.embedding_distance_threshold = embedding_distance_threshold
        self.semantic_weight = semantic_weight
    
    def retrieve(
        self,
        query: str,
        library,
        top_k: int = 10,
    ) -> List[Dict]:
        """Retrieve documents using hybrid search."""
        try:
            from llmware.retrieval import Query
        except ImportError:
            raise ImportError("llmware.retrieval.Query required for HybridRetriever")
        
        query_obj = Query(library)
        results = query_obj.hybrid_search(
            query,
            result_count=top_k,
            embedding_distance_threshold=self.embedding_distance_threshold,
            weights=[self.semantic_weight, 1.0 - self.semantic_weight],  # [semantic, text]
        )
        
        # Convert llmware results to standard format
        retrieved = []
        for result in results:
            retrieved.append({
                "doc_id": result.get("doc_ID", result.get("block_ID", "")),
                "score": result.get("score", 0.0),
                "text": result.get("text", ""),
                "metadata": result.get("metadata", {}),
            })
        
        return retrieved
