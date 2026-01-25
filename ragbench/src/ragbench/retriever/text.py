from typing import List, Dict

from .base import Retriever


class TextRetriever(Retriever):
    """Text-based retriever using llmware Query.text_search."""
    
    def __init__(self):
        """Initialize text retriever."""
        super().__init__()
    
    def retrieve(
        self,
        query: str,
        library,
        top_k: int = 10,
    ) -> List[Dict]:
        """Retrieve documents using text search."""
        try:
            from llmware.retrieval import Query
        except ImportError:
            raise ImportError("llmware.retrieval.Query required for TextRetriever")
        
        query_obj = Query(library)
        results = query_obj.text_search(
            query,
            result_count=top_k,
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
