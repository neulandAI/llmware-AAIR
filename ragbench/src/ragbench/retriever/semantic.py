from typing import List, Dict

from .base import Retriever


class SemanticRetriever(Retriever):
    """Semantic retriever using llmware Query.semantic_query."""
    
    def __init__(self, embedding_distance_threshold: float = 1.0):
        """
        Initialize semantic retriever.
        
        Args:
            embedding_distance_threshold: Maximum embedding distance threshold
        """
        super().__init__()
        self.embedding_distance_threshold = embedding_distance_threshold
    
    def retrieve(
        self,
        query: str,
        library,
        top_k: int = 10,
    ) -> List[Dict]:
        """Retrieve documents using semantic search."""
        try:
            from llmware.retrieval import Query
        except ImportError:
            raise ImportError("llmware.retrieval.Query required for SemanticRetriever")
        
        query_obj = Query(library)
        results = query_obj.semantic_query(
            query,
            result_count=top_k,
            embedding_distance_threshold=self.embedding_distance_threshold,
        )
        
        # Convert llmware results to standard format
        retrieved = []
        for result in results:
            retrieved.append({
                "doc_id": result.get("doc_ID", result.get("block_ID", "")),
                "score": 1.0 - result.get("similarity", 0.0),  # Convert similarity to distance-like score
                "text": result.get("text", ""),
                "metadata": result.get("metadata", {}),
            })
        
        return retrieved
