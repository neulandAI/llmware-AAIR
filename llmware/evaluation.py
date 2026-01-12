import json
import logging
import math
import time
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel

from llmware.retrieval import Query

logger = logging.getLogger(__name__)


class ResourcePage(BaseModel):
    document: str
    physical_page: List[int]
    embedded_page: List[int]


class QAItem(BaseModel):
    query: str
    answer: str
    source_docs: List[ResourcePage]


class EvaluationDataset(BaseModel):
    data: List[QAItem]


class EvaluationResult(BaseModel):
    """Results for a single query."""
    query: str
    latency_ms: float
    hit: bool
    reciprocal_rank: float
    recall: float
    ndcg: float
    retrieved_docs: List[str]
    relevant_docs: List[str]


class EvaluationMetrics(BaseModel):
    """Aggregate metrics across all queries."""
    total_queries: int
    k: int
    avg_latency_ms: float
    hit_rate: float
    mrr: float
    recall_at_k: float
    ndcg_at_k: float
    per_query: List[EvaluationResult]


class Evaluator:
    """
    Evaluates retrieval quality against a QA dataset.

    """

    def __init__(self, library, dataset_path: str):
        self.library = library
        self.query = Query(self.library)
        self.dataset_path = dataset_path

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
        """Build set of document names that are relevant."""
        return {doc.document for doc in source_docs}

    def _build_relevant_pages(self, source_docs: List[ResourcePage]) -> set:
        """Build set of (document, page) tuples that are relevant."""
        relevant = set()
        for doc in source_docs:
            for page in doc.physical_page:
                relevant.add((doc.document, page))
        return relevant

    def _extract_retrieved_docs(self, results: List[Dict], top_k: int) -> List[str]:
        """Extract file_source from query results."""
        return [r.get("file_source", "") for r in results[:top_k]]

    def _extract_retrieved_pages(self, results: List[Dict], top_k: int) -> List[tuple]:
        """Extract (file_source, page_num) from query results."""
        retrieved = []
        for r in results[:top_k]:
            file_source = r.get("file_source", "")
            page_num = r.get("page_num", 0)
            retrieved.append((file_source, page_num))
        return retrieved

    def _calculate_hit(self, retrieved: List, relevant: set) -> bool:
        """Hit = at least one relevant item in retrieved."""
        return any(r in relevant for r in retrieved)

    def _calculate_reciprocal_rank(self, retrieved: List, relevant: set) -> float:
        """RR = 1/rank of first relevant result."""
        for i, r in enumerate(retrieved):
            if r in relevant:
                return 1.0 / (i + 1)
        return 0.0

    def _calculate_recall(self, retrieved: List, relevant: set) -> float:
        """Recall@k = |retrieved ∩ relevant| / |relevant|"""
        if not relevant:
            return 0.0
        found = len(set(retrieved) & relevant)
        return found / len(relevant)

    def _calculate_ndcg(self, retrieved: List, relevant: set, k: int) -> float:
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

    def _run_metrics(self, retrieved: List, relevant: set, k: int) -> Dict:
        """Calculate all metrics for a single query."""
        return {
            "hit": self._calculate_hit(retrieved, relevant),
            "rr": self._calculate_reciprocal_rank(retrieved, relevant),
            "recall": self._calculate_recall(retrieved, relevant),
            "ndcg": self._calculate_ndcg(retrieved, relevant, k),
        }

    def _aggregate_results(
        self, 
        per_query_results: List[EvaluationResult], 
        latencies: List[float],
        hits: List[int],
        rrs: List[float],
        recalls: List[float],
        ndcgs: List[float],
        top_k: int
    ) -> EvaluationMetrics:
        n = len(per_query_results)
        return EvaluationMetrics(
            total_queries=n,
            k=top_k,
            avg_latency_ms=sum(latencies) / n if n else 0,
            hit_rate=sum(hits) / n if n else 0,
            mrr=sum(rrs) / n if n else 0,
            recall_at_k=sum(recalls) / n if n else 0,
            ndcg_at_k=sum(ndcgs) / n if n else 0,
            per_query=per_query_results
        )

    def evaluate(self, top_k: int = 10, by_page: bool = True, verbose: bool = False) -> EvaluationMetrics:
        """
        Run evaluation on the dataset.
        
        Args:
            top_k: Number of results to retrieve per query.
            by_page: If True, match by (document, page). If False, match by document only.
            verbose: If True, print detailed per-query results.
            
        Returns:
            EvaluationMetrics with NDCG@k, Recall@k, Hit Rate, MRR, latency.
        """
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

            metrics = self._run_metrics(retrieved, relevant, top_k)

            hits.append(1 if metrics["hit"] else 0)
            rrs.append(metrics["rr"])
            recalls.append(metrics["recall"])
            ndcgs.append(metrics["ndcg"])

            retrieved_strs = [str(r) for r in retrieved]
            relevant_strs = [str(r) for r in relevant]

            if verbose:
                print(f"\n{'='*60}")
                print(f"Query {i+1}: {item.query}")
                print(f"Hit: {metrics['hit']} | RR: {metrics['rr']:.2f} | Recall: {metrics['recall']:.2f} | NDCG: {metrics['ndcg']:.2f}")
                print(f"Expected:  {relevant_strs}")
                print(f"Retrieved: {retrieved_strs}")

            per_query_results.append(EvaluationResult(
                query=item.query,
                latency_ms=latency_ms,
                hit=metrics["hit"],
                reciprocal_rank=metrics["rr"],
                recall=metrics["recall"],
                ndcg=metrics["ndcg"],
                retrieved_docs=retrieved_strs,
                relevant_docs=relevant_strs
            ))

        if verbose:
            print(f"\n{'='*60}\n")

        return self._aggregate_results(
            per_query_results, latencies, hits, rrs, recalls, ndcgs, top_k
        )