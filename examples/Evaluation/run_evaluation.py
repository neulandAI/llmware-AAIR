from llmware.library import Library
from llmware.evaluation import Evaluator

DOCUMENTS_PATH = ""
DATASET_PATH = ""
LIBRARY_NAME = ""
TOP_K = 10
VERBOSE = True
METRICS = ["hit_rate", "mrr", "recall", "ndcg"]

library = Library().create_new_library(LIBRARY_NAME)
library.add_files(DOCUMENTS_PATH)
library.install_new_embedding(embedding_model_name="mini-lm-sbert", vector_db="chromadb")

evaluator = Evaluator.create("retrieval", library, DATASET_PATH, metrics=METRICS)
results = evaluator.evaluate(top_k=TOP_K, by_page=True, verbose=VERBOSE)

print(f"Hit Rate: {results.hit_rate:.3f}")
print(f"MRR: {results.mrr:.3f}")
print(f"Recall@{TOP_K}: {results.recall_at_k:.3f}")
print(f"NDCG@{TOP_K}: {results.ndcg_at_k:.3f}")
print(f"Latency: {results.avg_latency_ms:.1f}ms")
