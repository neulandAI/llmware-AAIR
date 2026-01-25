# RAGBench

A research codebase for systematic ablation studies of RAG (Retrieval-Augmented Generation) pipeline components.

## Overview

RAGBench provides a modular framework to test and compare different components of the RAG pipeline:
- **Parsers**: Document parsing (PDF, DOCX, etc.)
- **Chunkers**: Text chunking strategies
- **Encoders**: Embedding models
- **Retrievers**: Retrieval strategies (semantic, text, hybrid)
- **Rerankers**: Reranking models
- **Generators**: Response generation models

## Installation

```bash
# Install dependencies
pip install -e .
```

## Quick Start

1. **Prepare your dataset** in JSON format:

```json
{
  "queries": [
    {"id": "q1", "text": "What is machine learning?"},
    {"id": "q2", "text": "Explain neural networks"}
  ],
  "corpus": {
    "doc1": {
      "text": "Machine learning is a subset of artificial intelligence...",
      "title": "ML Introduction"
    },
    "doc2": {
      "text": "Neural networks are computing systems inspired by...",
      "title": "Neural Networks"
    }
  },
  "ground_truth": {
    "q1": ["doc1"],
    "q2": ["doc2"]
  }
}
```

2. **Run experiments**:

```python
from ragbench.runner import Runner
from ragbench.parser import UnstructuredParser
from ragbench.chunker import FixedSizeChunker
from ragbench.encoder import QwenEmbedding4B
from ragbench.retriever import SemanticRetriever
from ragbench.datasets import CustomDataset

config = {"dataset_path": "data/dataset.json"}
dataset = CustomDataset(config["dataset_path"])
runner = Runner(config)

result = runner.test(
    parser=UnstructuredParser(),
    chunker=FixedSizeChunker(chunk_size=500),
    encoder=QwenEmbedding4B(),
    retriever=SemanticRetriever(),
    dataset=dataset,
    top_k=10,
)

print(result.metrics)
```

3. **Run sweeps**:

Edit `main.py` to configure your sweep and run:

```bash
python main.py
```

## Architecture

The codebase follows a simple, modular design:

- **Base classes**: Each component type has an ABC base class
- **Simple initialization**: Components are initialized once per run via `__init__()`
- **Dict-based configs**: No complex schemas, just simple dicts
- **Sweep-friendly**: Easy to configure and run parameter sweeps

## Components

### Parsers
- `UnstructuredParser`
- `LlmwareParser`
- `RagFlowParser`
- `DoclingParser`
- `MarkerParser`
- And more...

### Chunkers
- `FixedSizeChunker`: Fixed-size chunks with optional overlap

### Encoders
- `QwenEmbedding4B`
- `QwenEmbedding8B`
- `Voyage3Large`
- `OctenEmbedding4B`
- `OctenEmbedding8B`

### Retrievers
- `SemanticRetriever`: Semantic search using llmware
- `TextRetriever`: Text-based search
- `HybridRetriever`: Hybrid semantic + text search

### Rerankers
- `CrossEncoderReranker`: Cross-encoder reranking

### Generators
- `LLMGenerator`: LLM-based response generation

## Evaluation Metrics

The framework supports retrieval evaluation metrics:
- Precision@K
- Recall@K
- MRR (Mean Reciprocal Rank)
- NDCG@K (Normalized Discounted Cumulative Gain)

## Adding New Components

To add a new component, inherit from the base class:

```python
from ragbench.chunker.base import Chunker

class MyChunker(Chunker):
    def __init__(self, param1=100):
        super().__init__()
        self.param1 = param1
    
    def chunk(self, documents):
        # Your chunking logic
        return chunks
```

## Results

Results are saved as JSON files with:
- Configuration (component names and parameters)
- Metrics (evaluation scores)
- Metadata (timestamps, dataset info)

## Dependencies

- `llmware`: Core RAG library
- `numpy`: Numerical operations
- `scikit-learn`: Evaluation metrics
- `sentence-transformers`: Reranking models
- `tqdm`: Progress bars

## License

[Add your license here]
