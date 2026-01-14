import json
import time
from pathlib import Path
from typing import List, Dict, Optional

from llmware.retrieval import Query

from .models import (
    ResourcePage,
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


AVAILABLE_METRICS = ["hit_rate", "mrr", "recall", "ndcg"]


class MetricNotFoundError(Exception):
    """Raised when a requested metric does not exist."""
    pass


class RetrievalEvaluator:
    """Evaluates retrieval quality against a QA dataset."""

    name = "retrieval"

    def __init__(self, library, dataset_path: str, metrics: Optional[List[str]] = None):
        """
        Args:
            library: Library object with embeddings installed
            dataset_path: Path to QA dataset JSON
            metrics: List of metrics to compute. If None, computes all.
                     Options: hit_rate, mrr, recall, ndcg
        """
        self.library = library
        self.query = Query(self.library)
        self.dataset_path = dataset_path
        
        if metrics is None:
            self.metrics = AVAILABLE_METRICS
        else:
            invalid = [m for m in metrics if m not in AVAILABLE_METRICS]
            if invalid:
                raise MetricNotFoundError(
                    f"Metrics not found: {invalid}. Available: {AVAILABLE_METRICS}"
                )
            self.metrics = metrics

    def _load_dataset(self) -> EvaluationDataset:
        json_path = Path(self.dataset_path)

        if not json_path.exists():
            raise FileNotFoundError(f"Dataset not found: {json_path}")

        with json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            raw = {"data": raw}

        return EvaluationDataset.model_validate(raw)

    def _build_relevant_docs(self, source_docs: List[ResourcePage]) -> set:
        return {doc.document for doc in source_docs}

    def _build_relevant_pages(self, source_docs: List[ResourcePage]) -> set:
        relevant = set()
        for doc in source_docs:
            for page in doc.physical_page:
                relevant.add((doc.document, page))
        return relevant

    def _extract_retrieved_docs(self, results: List[Dict], top_k: int) -> List[str]:
        return [r.get("file_source", "") for r in results[:top_k]]

    def _extract_retrieved_pages(self, results: List[Dict], top_k: int) -> List[tuple]:
        retrieved = []
        for r in results[:top_k]:
            file_source = r.get("file_source", "")
            page_num = r.get("page_num", 0)
            retrieved.append((file_source, page_num))
        return retrieved

    def evaluate(self, top_k: int = 10, by_page: bool = True, verbose: bool = False) -> EvaluationMetrics:
        dataset = self._load_dataset()

        per_query_results = []
        hits = []
        rrs = []
        recalls = []
        ndcgs = []
        latencies = []

        for i, item in enumerate(dataset.data):
            start = time.time()
            results = self.query.semantic_query(item.query, result_count=top_k)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

            if by_page:
                relevant = self._build_relevant_pages(item.source_docs)
                retrieved = self._extract_retrieved_pages(results, top_k)
            else:
                relevant = self._build_relevant_docs(item.source_docs)
                retrieved = self._extract_retrieved_docs(results, top_k)

            hit = calculate_hit(retrieved, relevant) if "hit_rate" in self.metrics else False
            rr = calculate_reciprocal_rank(retrieved, relevant) if "mrr" in self.metrics else 0.0
            recall = calculate_recall(retrieved, relevant) if "recall" in self.metrics else 0.0
            ndcg = calculate_ndcg(retrieved, relevant, top_k) if "ndcg" in self.metrics else 0.0

            hits.append(1 if hit else 0)
            rrs.append(rr)
            recalls.append(recall)
            ndcgs.append(ndcg)

            retrieved_strs = [str(r) for r in retrieved]
            relevant_strs = [str(r) for r in relevant]

            if verbose:
                print(f"\n{'='*60}")
                print(f"Query {i+1}: {item.query}")
                parts = []
                if "hit_rate" in self.metrics:
                    parts.append(f"Hit: {hit}")
                if "mrr" in self.metrics:
                    parts.append(f"RR: {rr:.2f}")
                if "recall" in self.metrics:
                    parts.append(f"Recall: {recall:.2f}")
                if "ndcg" in self.metrics:
                    parts.append(f"NDCG: {ndcg:.2f}")
                print(" | ".join(parts))
                print(f"Expected:  {relevant_strs}")
                print(f"Retrieved: {retrieved_strs}")

            per_query_results.append(EvaluationResult(
                query=item.query,
                latency_ms=latency_ms,
                hit=hit,
                reciprocal_rank=rr,
                recall=recall,
                ndcg=ndcg,
                retrieved_docs=retrieved_strs,
                relevant_docs=relevant_strs
            ))

        if verbose:
            print(f"\n{'='*60}\n")

        n = len(per_query_results)
        return EvaluationMetrics(
            total_queries=n,
            k=top_k,
            avg_latency_ms=sum(latencies) / n if n else 0,
            hit_rate=sum(hits) / n if n and "hit_rate" in self.metrics else 0,
            mrr=sum(rrs) / n if n and "mrr" in self.metrics else 0,
            recall_at_k=sum(recalls) / n if n and "recall" in self.metrics else 0,
            ndcg_at_k=sum(ndcgs) / n if n and "ndcg" in self.metrics else 0,
            per_query=per_query_results
        )
