from llmware.library import Library
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..config import Config
from ..evaluation import evaluate_retrieval
from ..results import ExperimentResult, save_result
from ..parser.base import Parser
from ..chunker.base import Chunker
from ..encoder.base import Encoder
from ..retriever.base import Retriever
from ..datasets.base import Dataset
from ..reranker.base import Reranker
from ..generator.base import Generator


class Runner:
    def __init__(
        self,
        config: Config,
        parser: Parser,
        chunker: Chunker,
        encoder: Encoder,
        retriever: Retriever,
        dataset: Dataset,
        reranker: Reranker,
        generator: Generator,
    ):
        """
        Initialize runner with all components.
        
        Args:
            config: Configuration dataclass with dataset and other settings
            parser: Parser instance
            chunker: Chunker instance
            encoder: Encoder instance
            retriever: Retriever instance
            dataset: Dataset instance
            reranker: Reranker instance (use NoOpReranker if not needed)
            generator: Generator instance (use NoOpGenerator if not needed)
        """
        self.config: Config = config
        self.parser: Parser = parser
        self.chunker: Chunker = chunker
        self.encoder: Encoder = encoder
        self.retriever: Retriever = retriever
        self.dataset: Dataset = dataset
        self.reranker: Reranker = reranker
        self.generator: Generator = generator
        self.library: Optional[Library] = None
        
        # Create results directory
        Path(self.config.results_path).parent.mkdir(parents=True, exist_ok=True)

    def prepare_llmware_lib(self, documents: List[Dict], encoder: Encoder) -> None:
        """
        Prepare llmware library with parsed and chunked documents.
        
        Args:
            documents: List of document dicts with "text" and "doc_id"
            encoder: Encoder instance for embedding
        """
        import tempfile
        import os
        from pathlib import Path
        
        # Create temporary directory for documents
        temp_dir = tempfile.mkdtemp()
        
        # Write documents to files (llmware expects files)
        doc_files = []
        for doc in documents:
            doc_id = doc.get("doc_id", "")
            text = doc.get("text", "")
            
            # Create a temporary file for this document
            file_path = os.path.join(temp_dir, f"{doc_id}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
            doc_files.append(file_path)
        
        # Create library and add documents
        self.library = Library().create_new_library("ragbench_lib")
        self.library.add_files(temp_dir)
        
        # Embed documents using encoder
        embedding_model_name = encoder.get_model_name()
        try:
            # Try llmware embedding interface
            self.library.install_new_embedding(embedding_model_name)
        except Exception:
            # If llmware doesn't support this model name, use encoder directly
            # This is a fallback - in practice, you might need to adjust model names
            # to match what llmware expects
            try:
                self.library.install_new_embedding(embedding_model_name)
            except Exception as e:
                raise ValueError(f"Failed to install embedding {embedding_model_name}: {e}. "
                               f"You may need to use a model name that llmware recognizes, "
                               f"or implement custom embedding logic.")

    def run(self) -> ExperimentResult:
        """
        Run the full pipeline for one compbination of components.
        
        Returns:
            ExperimentResult with metrics
        """
        # Load dataset
        data = self.dataset.load()
        queries: List[Dict] = data["queries"]
        corpus: Dict[str, Dict] = data["corpus"]
        ground_truth: Dict[str, set] = data["ground_truth"]
        
        # Step 1: Parse documents
        parsed_docs: List[Dict] = []
        for doc_id, doc_data in corpus.items():
            # Get text from corpus (assumes corpus already has text)
            # If file_path is provided and parser supports it, parse the file
            # Otherwise, use text directly
            text = doc_data.get("text", "")
            file_path = doc_data.get("file_path")
            
            # If file_path exists and parser can handle it, parse the file
            if file_path and hasattr(self.parser, "parse_one"):
                try:
                    sample = {
                        "file_path": file_path,
                        "text": text,  # Fallback text
                        "doc_id": doc_id,
                    }
                    parsed_text = self.parser.parse_one(sample)
                except Exception:
                    # If parsing fails, use text directly
                    parsed_text = text
            else:
                # Use text directly
                parsed_text = text
            
            parsed_docs.append({
                "doc_id": doc_id,
                "text": parsed_text,
                "metadata": doc_data.get("metadata", {}),
            })
        
        # Step 2: Chunk documents
        chunks: List[Dict] = self.chunker.chunk(parsed_docs)
        
        # Step 3: Prepare library and embed
        self.prepare_llmware_lib(chunks, self.encoder)
        
        # Step 4: Retrieve and evaluate for each query
        all_metrics: List[Dict[str, float]] = []
        
        for query in queries:
            query_id = query.get("id", "")
            query_text = query.get("text", "")
            
            # Retrieve
            retrieved: List[Dict] = self.retriever.retrieve(query_text, self.library, top_k=self.config.top_k)
            retrieved_doc_ids: List[str] = [r["doc_id"] for r in retrieved]
            
            # Rerank (always use reranker, even if it's NoOpReranker)
            retrieved = self.reranker.rerank(query_text, retrieved)
            retrieved_doc_ids = [r["doc_id"] for r in retrieved]
            
            # Get ground truth
            relevant_doc_ids: set = ground_truth.get(query_id, set())
            
            # Evaluate
            metrics: Dict[str, float] = evaluate_retrieval(retrieved_doc_ids, relevant_doc_ids)
            all_metrics.append(metrics)
        
        # Aggregate metrics across queries
        aggregated_metrics: Dict[str, float] = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                aggregated_metrics[key] = sum(values) / len(values)
        
        # Create result
        result_config: Dict[str, Any] = {
            "parser": self.parser.__class__.__name__,
            "chunker": self.chunker.__class__.__name__,
            "encoder": self.encoder.get_model_name(),
            "retriever": self.retriever.__class__.__name__,
            "reranker": self.reranker.__class__.__name__,
            "generator": self.generator.__class__.__name__,
            "top_k": self.config.top_k,
        }
        
        result = ExperimentResult(
            config=result_config,
            metrics=aggregated_metrics,
            metadata={
                "num_queries": len(queries),
                "num_documents": len(corpus),
            },
        )
        
        # Save result
        save_result(result, self.config.results_path, append=True)
        
        return result
