"""RAG Evaluator for retrieval quality metrics.

Evaluates RAG retrieval using gold_answer containment - a passage is relevant
if it contains the normalized gold answer. This matches the evaluation approach
used in multi-hop QA datasets like 2WikiMultiHopQA, HotpotQA, and MuSiQue.

Metrics:
- Hit@K: At least one passage in top-K contains the gold answer
- nDCG@K: Normalized Discounted Cumulative Gain (ranking quality)
- Precision@K: Fraction of top-K passages containing gold answer
- MRR: Mean Reciprocal Rank of first relevant passage
- MAP: Mean Average Precision
"""

import re
import string
import numpy as np
from typing import List, Dict, Any, Optional


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison.
    
    Removes articles, punctuation, extra whitespace, and lowercases text.
    
    Args:
        s: Text to normalize
        
    Returns:
        Normalized text string
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())
    
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text: str) -> str:
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


class RAGEvaluator:
    """Evaluates RAG retrieval quality using gold_answer containment.
    
    A passage is considered relevant if it contains the normalized gold answer.
    This is the standard evaluation approach for multi-hop QA datasets.
    
    Args:
        retrieval_results: List of dicts with format:
            [{
                "question": "...",
                "sorted_passage": [...],  # Retrieved passages in ranked order
                "sorted_passage_scores": [...],  # Similarity scores (optional)
                "gold_answer": "..."
            }]
    
    Example:
        >>> evaluator = RAGEvaluator(retrieval_results)
        >>> metrics = evaluator.evaluate_all(k=3)
        >>> print(f"Hit@3: {metrics['hit@3']:.3f}")
    """
    
    def __init__(self, retrieval_results: List[Dict[str, Any]]):
        self.retrieval_results = retrieval_results
    
    def is_relevant(self, passage: str, gold_answer: str) -> bool:
        """Check if a passage is relevant (contains gold answer).
        
        Args:
            passage: Retrieved passage text
            gold_answer: Expected answer
            
        Returns:
            True if normalized gold answer is in normalized passage
        """
        if not passage or not gold_answer:
            return False
        p = normalize_answer(passage)
        g = normalize_answer(gold_answer)
        return g in p
    
    def hit_ratio(self, k: int = 3) -> float:
        """Compute Hit@K metric.
        
        Hit@K = proportion of queries where at least one of the top-K 
        retrieved passages contains the gold answer.
        
        Args:
            k: Number of top passages to consider
            
        Returns:
            Hit ratio (0.0 to 1.0)
        """
        hits = 0
        total = len(self.retrieval_results)
        
        for item in self.retrieval_results:
            gold = item.get("gold_answer", "")
            passages = item.get("sorted_passage", [])[:k]
            
            if any(self.is_relevant(p, gold) for p in passages):
                hits += 1
        
        return hits / total if total > 0 else 0.0
    
    def ndcg(self, k: int = 3) -> float:
        """Compute nDCG@K (Normalized Discounted Cumulative Gain).
        
        Measures ranking quality - higher score if relevant passages
        appear earlier in the ranked list.
        
        Args:
            k: Number of top passages to consider
            
        Returns:
            Average nDCG score (0.0 to 1.0)
        """
        def dcg(rels: List[int]) -> float:
            """Compute Discounted Cumulative Gain."""
            return sum(rel / np.log2(i + 2) for i, rel in enumerate(rels))
        
        scores = []
        for item in self.retrieval_results:
            gold = item.get("gold_answer", "")
            passages = item.get("sorted_passage", [])[:k]
            
            # Binary relevance: 1 if passage contains answer, 0 otherwise
            relevances = [1 if self.is_relevant(p, gold) else 0 for p in passages]
            
            # Pad with zeros if fewer than k passages
            while len(relevances) < k:
                relevances.append(0)
            
            # Ideal ranking: all relevant passages first
            ideal = sorted(relevances, reverse=True)
            
            idcg = dcg(ideal)
            if idcg == 0:
                # No relevant passages found
                scores.append(0.0)
            else:
                scores.append(dcg(relevances) / idcg)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def precision_at_k(self, k: int = 3) -> float:
        """Compute Precision@K.
        
        Precision@K = (number of relevant passages in top-K) / K
        
        Args:
            k: Number of top passages to consider
            
        Returns:
            Average precision (0.0 to 1.0)
        """
        precisions = []
        
        for item in self.retrieval_results:
            gold = item.get("gold_answer", "")
            passages = item.get("sorted_passage", [])[:k]
            
            relevant_count = sum(1 for p in passages if self.is_relevant(p, gold))
            precisions.append(relevant_count / k if k > 0 else 0.0)
        
        return sum(precisions) / len(precisions) if precisions else 0.0
    
    def mrr(self, k: Optional[int] = None) -> float:
        """Compute Mean Reciprocal Rank (MRR).
        
        MRR = average of 1/rank of first relevant passage
        
        Args:
            k: Optional limit on passages to consider
        
        Returns:
            MRR score (0.0 to 1.0)
        """
        reciprocal_ranks = []
        
        for item in self.retrieval_results:
            gold = item.get("gold_answer", "")
            passages = item.get("sorted_passage", [])
            if k:
                passages = passages[:k]
            
            # Find rank of first relevant passage
            rr = 0.0
            for rank, passage in enumerate(passages, start=1):
                if self.is_relevant(passage, gold):
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def map_score(self, k: Optional[int] = None) -> float:
        """Compute Mean Average Precision (MAP).
        
        MAP = average of average precision at each relevant hit
        
        Args:
            k: Optional limit on passages to consider
            
        Returns:
            MAP score (0.0 to 1.0)
        """
        avg_precisions = []
        
        for item in self.retrieval_results:
            gold = item.get("gold_answer", "")
            passages = item.get("sorted_passage", [])
            if k:
                passages = passages[:k]
            
            relevant_count = 0
            precision_sum = 0.0
            
            for i, passage in enumerate(passages):
                if self.is_relevant(passage, gold):
                    relevant_count += 1
                    precision_sum += relevant_count / (i + 1)
            
            ap = precision_sum / relevant_count if relevant_count > 0 else 0.0
            avg_precisions.append(ap)
        
        return sum(avg_precisions) / len(avg_precisions) if avg_precisions else 0.0
    
    def evaluate_all(self, k: int = 3) -> Dict[str, float]:
        """Compute all retrieval metrics.
        
        Args:
            k: Number of top passages for K-based metrics
            
        Returns:
            Dict with all metric scores
        """
        return {
            f"hit@{k}": self.hit_ratio(k),
            f"ndcg@{k}": self.ndcg(k),
            f"precision@{k}": self.precision_at_k(k),
            "mrr": self.mrr(k),
            "map": self.map_score(k)
        }

