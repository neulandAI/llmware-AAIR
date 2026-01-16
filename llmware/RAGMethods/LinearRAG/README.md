# LinearRAG

Graph-based RAG using Named Entity Recognition and Personalized PageRank for multi-hop retrieval.

## Installation

```bash
pip install igraph spacy pyarrow sentence-transformers tqdm
python -m spacy download en_core_web_trf
```

## Basic Usage

```python
from llmware.models import ModelCatalog
from llmware.RAGMethods.LinearRAG import LinearRAG, LinearRAGConfig
from llmware.RAGMethods.LinearRAG.adapters import SentenceTransformerAdapter, LLMWareModelWrapper

config = LinearRAGConfig(
    dataset_name="my_dataset",
    embedding_model=SentenceTransformerAdapter("all-mpnet-base-v2"),
    llm_model=LLMWareModelWrapper(ModelCatalog().load_model("llmware/bling-phi-3-gguf")),
    retrieval_top_k=5
)

rag = LinearRAG(config)
rag.index(["Einstein developed relativity in 1905.", "Einstein won the Nobel Prize in 1921."])

results = rag.retrieve([{"question": "When did Einstein win Nobel?", "answer": "1921"}])
results = rag.qa([{"question": "When did Einstein win Nobel?", "answer": "1921"}])  # with LLM
```

## Integration with llmware/evaluation

> **Note:** The `RetrievalEvaluator` in `llmware/evaluation` is designed for `llmware.Library` + `Query`. LinearRAG uses its own retrieval, so use the **metric functions directly** instead.

```python
from llmware.RAGMethods.LinearRAG import LinearRAG, LinearRAGConfig
from llmware.RAGMethods.LinearRAG.adapters import SentenceTransformerAdapter
from llmware.evaluation import (
    calculate_hit,
    calculate_reciprocal_rank,
    calculate_recall,
    calculate_ndcg,
)

# 1. Setup LinearRAG
config = LinearRAGConfig(
    dataset_name="eval_test",
    embedding_model=SentenceTransformerAdapter("all-mpnet-base-v2"),
    retrieval_top_k=10
)
rag = LinearRAG(config)
rag.index(passages)

# 2. Prepare questions with ground truth
# Ground truth format: list of relevant document identifiers per question
questions = [
    {"question": "When did Einstein win Nobel?", "answer": "1921"},
    {"question": "What theory did Einstein develop?", "answer": "relativity"},
]
ground_truth = [
    ["doc1.pdf", "doc2.pdf"],  # relevant docs for question 1
    ["doc3.pdf"],              # relevant docs for question 2
]

# 3. Run retrieval
results = rag.retrieve(questions)

# 4. Evaluate using llmware metrics
hits, rrs, recalls, ndcgs = [], [], [], []

for result, relevant_docs in zip(results, ground_truth):
    # Extract doc identifiers from passages (customize for your passage format)
    # Example: if passages contain "Source: doc1.pdf" 
    retrieved = []
    for passage in result["sorted_passage"]:
        # Your extraction logic here - depends on how you format passages
        doc_id = extract_doc_id(passage)  # implement based on your format
        retrieved.append(doc_id)
    
    relevant = set(relevant_docs)
    
    hits.append(calculate_hit(retrieved, relevant))
    rrs.append(calculate_reciprocal_rank(retrieved, relevant))
    recalls.append(calculate_recall(retrieved, relevant))
    ndcgs.append(calculate_ndcg(retrieved, relevant, k=10))

n = len(results)
print(f"Hit Rate: {sum(hits)/n:.3f}")
print(f"MRR: {sum(rrs)/n:.3f}")
print(f"Recall@10: {sum(recalls)/n:.3f}")
print(f"NDCG@10: {sum(ndcgs)/n:.3f}")
```

### Key Point: Passage Format

For evaluation to work, your indexed passages should include document identifiers. Format them when indexing:

```python
# When indexing, include source info in passages
passages = [
    "Source: einstein_bio.pdf | Page: 15 | Einstein won the Nobel Prize in 1921.",
    "Source: physics_history.pdf | Page: 42 | The theory of relativity was published in 1905.",
]
rag.index(passages)

# Then extract during evaluation
def extract_doc_id(passage):
    if "Source:" in passage:
        return passage.split("Source:")[1].split("|")[0].strip()
    return passage[:50]  # fallback
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_name` | *required* | Storage folder name |
| `embedding_model` | None | Embedding model (`SentenceTransformerAdapter`) |
| `llm_model` | None | LLM for QA (`LLMWareModelWrapper`) |
| `working_dir` | `"./linearrag_data"` | Base storage directory |
| `retrieval_top_k` | 3 | Number of passages to retrieve |
| `damping` | 0.5 | PageRank damping factor |

## Adapters

| Type | Adapter | Usage |
|------|---------|-------|
| Embedding | `SentenceTransformerAdapter` | `SentenceTransformerAdapter("all-mpnet-base-v2")` |
| Embedding | `LLMWareEmbeddingAdapter` | `LLMWareEmbeddingAdapter(llmware_model)` |
| LLM | `LLMWareModelWrapper` | `LLMWareModelWrapper(ModelCatalog().load_model(...))` |
| LLM | `OpenAILLMAdapter` | `OpenAILLMAdapter(api_key, model="gpt-4o-mini")` |
| LLM | `HuggingFaceLLMAdapter` | `HuggingFaceLLMAdapter(model_name, device="cuda")` |

## Output Format

```python
# rag.retrieve() returns:
[{"question": "...", "sorted_passage": ["p1", "p2"], "sorted_passage_scores": [0.9, 0.7], "gold_answer": "..."}]

# rag.qa() adds:
[{..., "pred_answer": "LLM generated answer"}]
```
