"""Adapter for making LinearRAG compatible with RetrievalEvaluator.

This adapter provides a Query-compatible interface so LinearRAG can be
used with the existing evaluation framework without modifying it.
"""

from typing import List, Dict, Any


class LinearRAGQueryAdapter:
    """Adapter that makes LinearRAG compatible with RetrievalEvaluator.
    
    This adapter provides a `semantic_query()` method that matches the
    interface expected by `RetrievalEvaluator`, allowing LinearRAG to be
    evaluated using the same metrics and dataset format.
    
    Args:
        linear_rag: Initialized LinearRAG instance with indexed passages.
                    Passages must have been indexed with metadata (file_source, page_num)
                    for evaluation to work correctly.
    
    Example:
        >>> from llmware.RAGMethods.LinearRAG import LinearRAG, LinearRAGConfig, LinearRAGQueryAdapter
        >>> from llmware.evaluation import RetrievalEvaluator
        >>> 
        >>> # Setup LinearRAG with metadata
        >>> config = LinearRAGConfig(dataset_name="my_data", ...)
        >>> rag = LinearRAG(config)
        >>> passages = [
        ...     {"text": "passage text", "file_source": "doc.pdf", "page_num": 1},
        ...     ...
        ... ]
        >>> rag.index(passages)
        >>> 
        >>> # Create adapter and evaluator
        >>> adapter = LinearRAGQueryAdapter(rag)
        >>> evaluator = RetrievalEvaluator.__new__(RetrievalEvaluator)
        >>> evaluator.query = adapter
        >>> evaluator.dataset_path = "qa_dataset.json"
        >>> evaluator.metrics = ["hit_rate", "mrr", "recall", "ndcg"]
        >>> 
        >>> # Evaluate
        >>> results = evaluator.evaluate(top_k=10)
        >>> print(f"MRR: {results.mrr:.3f}")
    """
    
    def __init__(self, linear_rag: Any):
        """
        Args:
            linear_rag: LinearRAG instance with indexed passages containing metadata
        """
        self.linear_rag = linear_rag
    
    def semantic_query(self, query: str, result_count: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic query matching the Query.semantic_query interface.
        
        This method wraps LinearRAG's retrieve() method and transforms the output
        to match the format expected by RetrievalEvaluator.
        
        Args:
            query: The question/query text
            result_count: Number of results to return
            
        Returns:
            List of dicts with 'file_source' and 'page_num' keys, matching
            the format returned by Query.semantic_query()
        """
        # Store and temporarily override retrieval_top_k
        original_top_k = self.linear_rag.config.retrieval_top_k
        self.linear_rag.config.retrieval_top_k = result_count
        
        try:
            # Call LinearRAG retrieve
            results = self.linear_rag.retrieve([{"question": query, "answer": ""}])
            
            # Transform to Query.semantic_query format
            if results and results[0].get("sorted_passage_metadata"):
                return [
                    {
                        "file_source": m.get("file_source", ""),
                        "page_num": m.get("page_num", 0)
                    }
                    for m in results[0]["sorted_passage_metadata"]
                ]
            return []
        finally:
            # Restore original top_k
            self.linear_rag.config.retrieval_top_k = original_top_k

