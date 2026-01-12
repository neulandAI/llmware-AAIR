from typing import List
from pydantic import BaseModel


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
    query: str
    latency_ms: float
    hit: bool
    reciprocal_rank: float
    recall: float
    ndcg: float
    retrieved_docs: List[str]
    relevant_docs: List[str]


class EvaluationMetrics(BaseModel):
    total_queries: int
    k: int
    avg_latency_ms: float
    hit_rate: float
    mrr: float
    recall_at_k: float
    ndcg_at_k: float
    per_query: List[EvaluationResult]
