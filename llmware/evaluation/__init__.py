from typing import List, Dict, Any

from .models import (
    ResourcePage,
    QAItem,
    EvaluationDataset,
    EvaluationResult,
    EvaluationMetrics,
)
from .metrics import (
    calculate_hit,
    calculate_reciprocal_rank,
    calculate_recall,
    calculate_ndcg,
)
from .retrieval import RetrievalEvaluator, MetricNotFoundError, AVAILABLE_METRICS
from .rag_evaluator import RAGEvaluator, normalize_answer
from .answer_evaluator import AnswerEvaluator


EVALUATOR_REGISTRY = {
    "retrieval": RetrievalEvaluator,
    "rag": RAGEvaluator,
    "answer": AnswerEvaluator,
}


class EvaluatorNotFoundError(Exception):
    """Raised when a requested evaluator does not exist."""
    pass


class Evaluator:
    """
    Factory for creating evaluators.
    
    Usage:
        evaluator = Evaluator.create("retrieval", library, dataset_path, metrics=["mrr", "ndcg"])
        results = evaluator.evaluate(top_k=10)
    """

    @staticmethod
    def available_evaluators() -> List[str]:
        """Return list of available evaluator names."""
        return list(EVALUATOR_REGISTRY.keys())

    @staticmethod
    def available_metrics() -> List[str]:
        """Return list of available metrics."""
        return AVAILABLE_METRICS

    @staticmethod
    def create(name: str, library, dataset_path: str, metrics: List[str] = None):
        """
        Create an evaluator by name.
        
        Args:
            name: Evaluator name (e.g., "retrieval")
            library: Library object
            dataset_path: Path to QA dataset JSON
            metrics: List of metrics to compute. Validates they exist.
                     Options: hit_rate, mrr, recall, ndcg
            
        Raises:
            EvaluatorNotFoundError: If evaluator name not in registry
            MetricNotFoundError: If metric not available
        """
        if name not in EVALUATOR_REGISTRY:
            raise EvaluatorNotFoundError(
                f"Evaluator '{name}' not found. Available: {list(EVALUATOR_REGISTRY.keys())}"
            )

        return EVALUATOR_REGISTRY[name](library, dataset_path, metrics=metrics)


__all__ = [
    "ResourcePage",
    "QAItem", 
    "EvaluationDataset",
    "EvaluationResult",
    "EvaluationMetrics",
    "RetrievalEvaluator",
    "RAGEvaluator",
    "AnswerEvaluator",
    "Evaluator",
    "EvaluatorNotFoundError",
    "MetricNotFoundError",
    "EVALUATOR_REGISTRY",
    "AVAILABLE_METRICS",
    "calculate_hit",
    "calculate_reciprocal_rank",
    "calculate_recall",
    "calculate_ndcg",
    "normalize_answer",
]
