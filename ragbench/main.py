import itertools
from functools import reduce
from operator import mul

from ragbench.config import (
    Config,
    ChunkerConfig,
    ParserConfig,
    EncoderConfig,
    RetrieverConfig,
    DatasetConfig,
)
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


def setup():
    """Setup function for initialization."""
    pass


def create_parser(config: Config, parser_type: str):
    """Create parser instance from config string."""
    parser_map = {
        "unstructured": UnstructuredParser,
        "llmware": LlmwareParser,
    }
    if parser_type not in parser_map:
        raise ValueError(f"Unknown parser type: {parser_type}")
    return parser_map[parser_type]()


def create_encoder(config: Config, encoder_type: str):
    """Create encoder instance from config string."""
    encoder_map = {
        "qwen_4b": QwenEmbedding4B,
        "qwen_8b": QwenEmbedding8B,
        "voyage_3_large": Voyage3Large,
        "octen_4b": OctenEmbedding4B,
        "octen_8b": OctenEmbedding8B,
    }
    if encoder_type not in encoder_map:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    return encoder_map[encoder_type]()


def create_retriever(config: Config, retriever_type: str):
    """Create retriever instance from config string."""
    retriever_map = {
        "semantic": SemanticRetriever,
        "text": TextRetriever,
        "hybrid": HybridRetriever,
    }
    if retriever_type not in retriever_map:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
    return retriever_map[retriever_type]()


def create_chunker(config: Config, chunk_size: int, overlap: int):
    """Create chunker instance from config values."""
    return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)


def create_dataset(config: Config, dataset_path: str):
    """Create dataset instance from config path."""
    return CustomDataset(dataset_path)


def build_sweep_config(config: Config):
    """Build sweep configuration from config object."""
    # Build datasets
    datasets = [create_dataset(config, path) for path in config.dataset.dataset_paths]
    
    # Build parsers
    parsers = [create_parser(config, pt) for pt in config.parser.parser_types]
    
    # Build chunkers - create all combinations of chunk_size and overlap
    chunkers = []
    for chunk_size in config.chunker.chunk_sizes:
        for overlap in config.chunker.overlaps:
            chunkers.append(create_chunker(config, chunk_size, overlap))
    
    # Build encoders
    encoders = [create_encoder(config, et) for et in config.encoder.encoder_types]
    
    # Build retrievers
    retrievers = [create_retriever(config, rt) for rt in config.retriever.retriever_types]
    
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
    
    config = Config(
        top_k=10,
        results_path="results/experiment_results.jsonl",
        chunker=ChunkerConfig(
            chunk_sizes=[500, 1000],
            overlaps=[50, 100],
        ),
        parser=ParserConfig(
            parser_types=["unstructured", "llmware"],
        ),
        encoder=EncoderConfig(
            encoder_types=["qwen_4b", "qwen_8b"],
        ),
        retriever=RetrieverConfig(
            retriever_types=["semantic", "text", "hybrid"],
        ),
        dataset=DatasetConfig(
            dataset_paths=["data/dataset.json"],
        ),
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
