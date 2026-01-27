import itertools
from functools import reduce
from operator import mul

from ragbench.config import Config
from ragbench.runner import Runner
from ragbench.parser import UnstructuredParser, LlmwareParser
from ragbench.encoder import (
    QwenEmbedding4B,
    QwenEmbedding8B,
    Voyage3Large,
    OctenEmbedding4B,
    OctenEmbedding8B,
)
from ragbench.chunker import FixedSizeChunker
from ragbench.retriever import SemanticRetriever, TextRetriever, HybridRetriever
from ragbench.datasets import CustomDataset
from ragbench.reranker import NoOpReranker
from ragbench.generator import NoOpGenerator


def setup():
    """Setup function for initialization."""
    pass


def build_sweep_config(config: Config):
    """Build sweep configuration from config object and constructors."""
    # Define component constructors directly (not in config)
    parsers = [
        UnstructuredParser(),
        LlmwareParser(),
    ]
    
    encoders = [
        QwenEmbedding4B(),
        QwenEmbedding8B(),
        # Voyage3Large(),  # Uncomment if API key available
        # OctenEmbedding4B(),  # Uncomment if available
        # OctenEmbedding8B(),  # Uncomment if available
    ]
    
    retrievers = [
        SemanticRetriever(),
        TextRetriever(),
        HybridRetriever(),
    ]
    
    # Build datasets using constructors directly
    datasets = [CustomDataset(path) for path in config.dataset_paths]
    
    # Build chunkers using constructors directly - create all combinations of chunk_size and overlap
    chunkers = []
    for chunk_size in config.chunk_sizes:
        for overlap in config.overlaps:
            chunkers.append(FixedSizeChunker(chunk_size=chunk_size, overlap=overlap))
    
    return {
        "datasets": datasets,
        "parsers": parsers,
        "chunkers": chunkers,
        "encoders": encoders,
        "retrievers": retrievers,
    }


def calculate_total_combinations(config: Config):
    """Calculate total experiment combinations from config."""
    sweep_config = build_sweep_config(config)
    sweep_values = list(sweep_config.values())
    return reduce(mul, (len(values) for values in sweep_values), 1)


def main():
    setup()
    
    # Configuration - only parameter values, component constructors are in main.py
    config = Config(
        top_k=10,
        results_path="results/experiment_results.jsonl",
        chunk_sizes=[500, 1000],
        overlaps=[50, 100],
        dataset_paths=["data/dataset.json"],
    )
    
    sweep_config = build_sweep_config(config)
    
    # Get sweep attributes and values
    sweep_attrib = list(sweep_config.keys())
    sweep_values = list(sweep_config.values())
    
    # Calculate total combinations
    total_combinations = calculate_total_combinations(config)
    
    print(f"Running {total_combinations} experiment combinations...")
    
    # Run sweep - create a new runner for each combination
    for idx, current_values in enumerate(itertools.product(*sweep_values), 1):
        current_params = dict(zip(sweep_attrib, current_values))
        print(f"\n[{idx}/{total_combinations}] Running experiment:")
        
        try:
            runner = Runner(
                config=config,
                dataset=current_params["datasets"],
                parser=current_params["parsers"],
                chunker=current_params["chunkers"],
                encoder=current_params["encoders"],
                retriever=current_params["retrievers"],
                reranker=NoOpReranker(),
                generator=NoOpGenerator(),
            )
            
            res = runner.test()
            print(f"Results: {res.metrics}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
