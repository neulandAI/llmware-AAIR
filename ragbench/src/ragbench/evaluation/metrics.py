"""Simple retrieval evaluation metrics."""

from typing import List, Set


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Precision@K.
    
    Args:
        retrieved: List of retrieved document IDs (ordered by relevance)
        relevant: Set of relevant document IDs
        k: Number of top results to consider
    
    Returns:
        Precision@K score
    """
    if k == 0:
        return 0.0
    
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant)
    return relevant_retrieved / len(top_k)


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Recall@K.
    
    Args:
        retrieved: List of retrieved document IDs (ordered by relevance)
        relevant: Set of relevant document IDs
        k: Number of top results to consider
    
    Returns:
        Recall@K score
    """
    if not relevant:
        return 0.0
    
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant)
    return relevant_retrieved / len(relevant)


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        retrieved: List of retrieved document IDs (ordered by relevance)
        relevant: Set of relevant document IDs
    
    Returns:
        MRR score (0.0 if no relevant document found)
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@K (NDCG@K).
    
    Uses sklearn's ndcg_score for calculation.
    
    Args:
        retrieved: List of retrieved document IDs (ordered by relevance)
        relevant: Set of relevant document IDs
        k: Number of top results to consider
    
    Returns:
        NDCG@K score
    """
    try:
        from sklearn.metrics import ndcg_score
    except ImportError:
        # Fallback to simple implementation if sklearn not available
        return _simple_ndcg_at_k(retrieved, relevant, k)
    
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    
    # Create binary relevance vector for retrieved docs
    y_true = [1.0 if doc_id in relevant else 0.0 for doc_id in top_k]
    
    # Create score vector (higher score for earlier positions)
    y_score = [1.0 / (i + 1) for i in range(len(top_k))]
    
    # Ideal DCG: all relevant docs first
    ideal_y_true = sorted(y_true, reverse=True)
    
    if sum(ideal_y_true) == 0:
        return 0.0
    
    dcg = ndcg_score([y_true], [y_score], k=k)
    ideal_dcg = ndcg_score([ideal_y_true], [y_score], k=k)
    
    if ideal_dcg == 0:
        return 0.0
    
    return dcg / ideal_dcg


def _simple_ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Simple NDCG implementation without sklearn dependency."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        if doc_id in relevant:
            relevance = 1.0
            dcg += relevance / __log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate ideal DCG
    num_relevant = min(len(relevant), k)
    ideal_dcg = sum(1.0 / __log2(i + 2) for i in range(num_relevant))
    
    if ideal_dcg == 0:
        return 0.0
    
    return dcg / ideal_dcg


def __log2(x: float) -> float:
    """Calculate log base 2."""
    import math
    return math.log2(x) if x > 0 else 1.0


def evaluate_retrieval(
    retrieved: List[str],
    relevant: Set[str],
    k_values: List[int] = [1, 3, 5, 10],
) -> dict:
    """
    Evaluate retrieval results with multiple metrics.
    
    Args:
        retrieved: List of retrieved document IDs (ordered by relevance)
        relevant: Set of relevant document IDs
        k_values: List of K values to evaluate
    
    Returns:
        Dictionary with metric scores
    """
    results = {}
    
    for k in k_values:
        results[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        results[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    
    results["mrr"] = mrr(retrieved, relevant)
    
    return results
