"""Test LinearRAG evaluation on 2WikiMultiHopQA dataset.

This script evaluates LinearRAG using standard RAG metrics:
- Hit@K: At least one passage contains the gold answer
- nDCG@K: Ranking quality metric
- Contain Acc: Predicted answer contains gold answer (requires QA)
- LLM Acc: LLM judges prediction correctness (requires QA + LLM)

Also tracks timing for indexing, retrieval, and generation.
Tests all 3 chunking strategies: original, fixed_size, sentence_based.
"""

import json
import re
import time
import tempfile
import shutil
from typing import List, Dict, Any
from tqdm import tqdm

# Import LinearRAG
from llmware.RAGMethods.LinearRAG import (
    LinearRAG,
    LinearRAGConfig,
    SentenceTransformerAdapter,
    HuggingFaceLLMAdapter,
    get_chunking_strategy,
    ChunkingConfig,
)

# Import evaluators
from llmware.evaluation.rag_evaluator import RAGEvaluator
from llmware.evaluation.answer_evaluator import AnswerEvaluator

# Default models
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Chunking strategies to test
CHUNKING_STRATEGIES = ["original", "fixed_size", "sentence_based"]


def load_chunks(path: str, limit: int = None) -> List[str]:
    """Load chunks from JSON file."""
    with open(path, 'r') as f:
        chunks = json.load(f)
    
    if limit:
        chunks = chunks[:limit]
    
    # Remove the "N:" prefix from each chunk
    cleaned_chunks = []
    for chunk in chunks:
        # Remove prefix like "0:", "27:", etc.
        cleaned = re.sub(r'^\d+:', '', chunk).strip()
        cleaned_chunks.append(cleaned)
    
    return cleaned_chunks


def load_questions(path: str, limit: int = None) -> List[dict]:
    """Load questions from JSON file."""
    with open(path, 'r') as f:
        questions = json.load(f)
    
    if limit:
        questions = questions[:limit]
    
    return questions


def run_evaluation_for_strategy(
    strategy_name: str,
    raw_chunks: List[str],
    questions: List[dict],
    embedding_model: Any,
    top_k: int,
    max_tokens: int = 300,
    fixed_overlap: int = 50,
    sentence_max_tokens: int = 350,
    enable_qa: bool = False,
    llm_model: Any = None
) -> Dict[str, Any]:
    """
    Run evaluation for a specific chunking strategy.
    
    Args:
        strategy_name: Name of chunking strategy ('original', 'fixed_size', 'sentence_based')
        raw_chunks: Original chunks before chunking
        questions: List of question dicts with 'question' and 'answer' keys
        embedding_model: Embedding model adapter
        top_k: Number of passages to retrieve
        max_tokens: Max tokens for fixed-size chunking (default: 300, matches Sonal's)
        fixed_overlap: Token overlap for fixed-size chunking (default: 50)
        sentence_max_tokens: Max tokens for sentence-based chunking (default: 350)
        enable_qa: Whether to run QA generation for Contain/LLM accuracy
        llm_model: LLM model for QA (required if enable_qa=True)
        
    Returns:
        Dict with evaluation metrics, chunking metadata, and timing
    """
    print(f"\n{'='*60}")
    print(f"STRATEGY: {strategy_name.upper()}")
    print("="*60)
    
    print(f"\nApplying {strategy_name} chunking (max_tokens={max_tokens}, overlap={fixed_overlap})...")
    chunk_config = ChunkingConfig(
        fixed_max_tokens=max_tokens,           # 300 tokens for fixed-size
        fixed_overlap_tokens=fixed_overlap,    # 50 token overlap
        sentences_per_chunk=4,
        sentence_overlap=1,
        max_chunk_tokens=sentence_max_tokens,  # 350 tokens for sentence-based
        min_chunk_tokens=30,
        embedding_model_name=DEFAULT_EMBEDDING_MODEL
    )
    strategy = get_chunking_strategy(strategy_name, chunk_config)
    chunks, chunk_metadata = strategy.chunk(raw_chunks)
    
    print(f"  Original chunks: {chunk_metadata['original_count']}")
    print(f"  Output chunks:   {chunk_metadata['output_count']}")
    print(f"  Expansion ratio: {chunk_metadata['expansion_ratio']}x")
    if 'statistics' in chunk_metadata:
        stats = chunk_metadata['statistics']
        print(f"  Token stats: min={stats.get('min_tokens', 0)}, "
              f"max={stats.get('max_tokens', 0)}, mean={stats.get('mean_tokens', 0)}")
    
    # Create fresh working directory for this strategy
    working_dir = tempfile.mkdtemp()
    
    try:
        # Initialize LinearRAG
        config = LinearRAGConfig(
            dataset_name=f"2wiki_{strategy_name}",
            embedding_model=embedding_model,
            llm_model=llm_model if enable_qa else None,
            working_dir=working_dir,
            retrieval_top_k=top_k,
            spacy_model="en_core_web_trf"
        )
        rag = LinearRAG(config)
        
        # Index chunks with timing
        print(f"\nIndexing {len(chunks)} chunks...")
        index_start = time.time()
        rag.index(chunks)
        index_time = time.time() - index_start
        print(f"Indexing complete ({index_time:.2f}s)")
        
        # Prepare questions for LinearRAG format
        qa_questions = [
            {"question": q["question"], "answer": q.get("answer", "")}
            for q in questions
        ]
        
        # Retrieve with timing
        print(f"\nRetrieving for {len(qa_questions)} questions...")
        retrieval_start = time.time()
        retrieval_results = rag.retrieve(qa_questions)
        retrieval_time = time.time() - retrieval_start
        print(f"Retrieval complete ({retrieval_time:.2f}s)")
        
        # Add gold_answer to results for evaluation
        for result, q in zip(retrieval_results, questions):
            result["gold_answer"] = q.get("answer", "")
        
        # Evaluate retrieval with RAGEvaluator
        print(f"\nComputing retrieval metrics...")
        rag_evaluator = RAGEvaluator(retrieval_results)
        retrieval_metrics = rag_evaluator.evaluate_all(k=top_k)
        
        # Optional: QA generation and answer evaluation
        generation_time = 0.0
        contain_accuracy = 0.0
        llm_accuracy = 0.0
        
        if enable_qa and llm_model is not None:
            print(f"\nGenerating answers...")
            generation_start = time.time()
            qa_results = rag.qa(qa_questions)
            generation_time = time.time() - generation_start
            print(f"Generation complete ({generation_time:.2f}s)")
            
            # Evaluate answers
            print(f"\nEvaluating answers...")
            answer_evaluator = AnswerEvaluator(qa_results, llm_model=llm_model)
            answer_metrics = answer_evaluator.evaluate(
                compute_llm=True,
                max_workers=2,
                show_progress=True
            )
            contain_accuracy = answer_metrics["contain_accuracy"]
            llm_accuracy = answer_metrics.get("llm_accuracy", 0.0)
        
        # Compile results
        results = {
            "strategy": strategy_name,
            "chunk_count": len(chunks),
            "expansion_ratio": chunk_metadata['expansion_ratio'],
            "questions_evaluated": len(questions),
            # Retrieval metrics
            f"hit@{top_k}": retrieval_metrics[f"hit@{top_k}"],
            f"ndcg@{top_k}": retrieval_metrics[f"ndcg@{top_k}"],
            f"precision@{top_k}": retrieval_metrics[f"precision@{top_k}"],
            "mrr": retrieval_metrics["mrr"],
            "map": retrieval_metrics["map"],
            # Answer metrics (if QA enabled)
            "contain_accuracy": contain_accuracy,
            "llm_accuracy": llm_accuracy,
            # Timing (in seconds)
            "index_time_s": round(index_time, 2),
            "retrieval_time_s": round(retrieval_time, 2),
            "generation_time_s": round(generation_time, 2),
        }
        
        # Print results for this strategy
        print(f"\n{'-'*40}")
        print(f"Results for {strategy_name}:")
        print(f"  Hit@{top_k}:          {results[f'hit@{top_k}']:.3f}")
        print(f"  nDCG@{top_k}:         {results[f'ndcg@{top_k}']:.3f}")
        print(f"  Precision@{top_k}:    {results[f'precision@{top_k}']:.3f}")
        print(f"  MRR:              {results['mrr']:.3f}")
        if enable_qa:
            print(f"  Contain Acc:      {results['contain_accuracy']:.3f}")
            print(f"  LLM Acc:          {results['llm_accuracy']:.3f}")
        print(f"  Index Time:       {results['index_time_s']}s")
        print(f"  Retrieval Time:   {results['retrieval_time_s']}s")
        if enable_qa:
            print(f"  Generation Time:  {results['generation_time_s']}s")
        
        return results
        
    finally:
        # Clean up working directory
        shutil.rmtree(working_dir, ignore_errors=True)


def main():
    # Paths
    chunks_path = "/Users/edgue/Documents/GitHub/master-thesis-sonal/traditional_rag_experiments/dataset/2wikimultihopQA/chunks.json"
    questions_path = "/Users/edgue/Documents/GitHub/master-thesis-sonal/traditional_rag_experiments/dataset/2wikimultihopQA/questions.json"
    
    # Parameters
    NUM_QUESTIONS = 100
    TOP_K = 3
    MAX_TOKENS = 300           # Fixed-size chunking: 300 tokens
    FIXED_OVERLAP = 50         # Fixed overlap: 50 tokens
    SENTENCE_MAX_TOKENS = 350  # Sentence-based max: 350 tokens
    ENABLE_QA = True
    
    print("=" * 80)
    print("2WikiMultiHopQA Evaluation with LinearRAG")
    print("Testing All Chunking Strategies")
    print("=" * 80)
    print(f"\nSettings: questions={NUM_QUESTIONS}, top_k={TOP_K}, max_tokens={MAX_TOKENS}, "
          f"overlap={FIXED_OVERLAP}, sentence_max={SENTENCE_MAX_TOKENS}, qa={ENABLE_QA}")
    
    # Load data
    print("\nLoading chunks...")
    raw_chunks = load_chunks(chunks_path)
    print(f"Loaded {len(raw_chunks)} raw chunks")
    
    print("\nLoading questions...")
    questions = load_questions(questions_path, limit=NUM_QUESTIONS)
    print(f"Loaded {len(questions)} questions")
    
    # Initialize models
    print(f"\nLoading embedding model: {DEFAULT_EMBEDDING_MODEL}")
    embedding_model = SentenceTransformerAdapter(DEFAULT_EMBEDDING_MODEL)
    
    llm_model = None
    if ENABLE_QA:
        print(f"Loading LLM: {DEFAULT_LLM_MODEL}")
        llm_model = HuggingFaceLLMAdapter(DEFAULT_LLM_MODEL)
    
    # Run evaluation for each chunking strategy
    all_results = []
    for strategy_name in CHUNKING_STRATEGIES:
        results = run_evaluation_for_strategy(
            strategy_name=strategy_name,
            raw_chunks=raw_chunks,
            questions=questions,
            embedding_model=embedding_model,
            top_k=TOP_K,
            max_tokens=MAX_TOKENS,
            fixed_overlap=FIXED_OVERLAP,
            sentence_max_tokens=SENTENCE_MAX_TOKENS,
            enable_qa=ENABLE_QA,
            llm_model=llm_model
        )
        all_results.append(results)
    
    # Print comparison summary
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    
    # Header
    if ENABLE_QA:
        header = f"{'Strategy':<15} {'Chunks':<8} {'Hit@'+str(TOP_K):<8} {'nDCG@'+str(TOP_K):<8} {'Contain':<9} {'LLM_Acc':<9} {'Index_s':<9} {'Retr_s':<8} {'Gen_s':<8}"
    else:
        header = f"{'Strategy':<15} {'Chunks':<8} {'Hit@'+str(TOP_K):<8} {'nDCG@'+str(TOP_K):<8} {'Prec@'+str(TOP_K):<9} {'MRR':<8} {'Index_s':<9} {'Retr_s':<8}"
    
    print(header)
    print("-" * 100)
    
    for r in all_results:
        if ENABLE_QA:
            row = (f"{r['strategy']:<15} {r['chunk_count']:<8} "
                   f"{r[f'hit@{TOP_K}']:<8.3f} {r[f'ndcg@{TOP_K}']:<8.3f} "
                   f"{r['contain_accuracy']:<9.3f} {r['llm_accuracy']:<9.3f} "
                   f"{r['index_time_s']:<9} {r['retrieval_time_s']:<8} {r['generation_time_s']:<8}")
        else:
            row = (f"{r['strategy']:<15} {r['chunk_count']:<8} "
                   f"{r[f'hit@{TOP_K}']:<8.3f} {r[f'ndcg@{TOP_K}']:<8.3f} "
                   f"{r[f'precision@{TOP_K}']:<9.3f} {r['mrr']:<8.3f} "
                   f"{r['index_time_s']:<9} {r['retrieval_time_s']:<8}")
        print(row)
    
    print("=" * 100)
    
    # Find best strategy for each metric
    print("\nBest by metric:")
    key_metrics = [f'hit@{TOP_K}', f'ndcg@{TOP_K}', 'mrr']
    if ENABLE_QA:
        key_metrics.extend(['contain_accuracy', 'llm_accuracy'])
    
    for metric in key_metrics:
        best = max(all_results, key=lambda x: x[metric])
        print(f"  {metric:<20}: {best['strategy']} ({best[metric]:.3f})")
    
    return all_results


if __name__ == "__main__":
    main()
