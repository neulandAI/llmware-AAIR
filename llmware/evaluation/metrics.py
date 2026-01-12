import math
from typing import List


def calculate_hit(retrieved: List, relevant: set) -> bool:
    """Hit = at least one relevant item in retrieved."""
    return any(r in relevant for r in retrieved)


def calculate_reciprocal_rank(retrieved: List, relevant: set) -> float:
    """RR = 1/rank of first relevant result."""
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def calculate_recall(retrieved: List, relevant: set) -> float:
    """Recall@k = |retrieved ∩ relevant| / |relevant|"""
    if not relevant:
        return 0.0
    found = len(set(retrieved) & relevant)
    return found / len(relevant)


def calculate_ndcg(retrieved: List, relevant: set, k: int) -> float:
    """NDCG@k with binary relevance."""
    if not relevant:
        return 0.0

    dcg = 0.0
    for i, r in enumerate(retrieved[:k]):
        rel = 1.0 if r in relevant else 0.0
        dcg += rel / math.log2(i + 2)

    ideal_rels = [1.0] * min(len(relevant), k)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0
