"""Answer Evaluator for QA quality metrics.

Evaluates QA answer quality using two approaches:
1. Contain Accuracy - Does the predicted answer contain the gold answer?
2. LLM Accuracy - LLM judges if prediction is correct (requires LLM model)

These metrics require QA generation (predicted answers), not just retrieval.
"""

from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .rag_evaluator import normalize_answer


class AnswerEvaluator:
    """Evaluates QA answer quality.
    
    Two evaluation modes:
    - Contain Accuracy: Fast, no LLM needed - checks if gold answer is in prediction
    - LLM Accuracy: Uses LLM to judge if prediction matches gold answer
    
    Args:
        results: List of dicts with format:
            [{
                "question": "...",
                "pred_answer": "...",  # Generated answer
                "gold_answer": "..."   # Expected answer
            }]
        llm_model: Optional LLM model with infer(messages) method for LLM accuracy
    
    Example:
        >>> evaluator = AnswerEvaluator(results)
        >>> metrics = evaluator.evaluate()
        >>> print(f"Contain Acc: {metrics['contain_accuracy']:.3f}")
    """
    
    def __init__(
        self, 
        results: List[Dict[str, Any]], 
        llm_model: Optional[Any] = None
    ):
        self.results = results
        self.llm_model = llm_model
    
    @staticmethod
    def calculate_contain(pred_answer: str, gold_answer: str) -> int:
        """Check if predicted answer contains the gold answer.
        
        Uses normalized text comparison (lowercase, no punctuation, no articles).
        
        Args:
            pred_answer: Generated/predicted answer
            gold_answer: Expected gold answer
            
        Returns:
            1 if gold answer is contained in prediction, 0 otherwise
        """
        if not pred_answer or not gold_answer:
            return 0
        if isinstance(pred_answer, str) and pred_answer.strip() == "":
            return 0
        if isinstance(gold_answer, str) and gold_answer.strip() == "":
            return 0
        
        s1 = normalize_answer(pred_answer)
        s2 = normalize_answer(gold_answer)
        
        return 1 if s2 in s1 else 0
    
    def calculate_llm_accuracy(self, pred_answer: str, gold_answer: str) -> float:
        """Use LLM to judge if prediction matches gold answer.
        
        The LLM evaluates if the prediction is correct based on:
        1. Contains key information from gold answer
        2. Is factually accurate
        3. Does not contradict gold answer
        
        Args:
            pred_answer: Generated/predicted answer
            gold_answer: Expected gold answer
            
        Returns:
            1.0 if LLM judges correct, 0.0 otherwise
            
        Raises:
            ValueError: If llm_model is not configured
        """
        if self.llm_model is None:
            raise ValueError("LLM model required for LLM accuracy. Set llm_model parameter.")
        
        system_prompt = "You are an expert evaluator."
        
        user_prompt = f"""Please evaluate if the generated answer is correct by comparing it with the gold answer.
Generated answer: {pred_answer}
Gold answer: {gold_answer}

The generated answer should be considered correct if it:
1. Contains the key information from the gold answer
2. Is factually accurate and consistent with the gold answer
3. Does not contain any contradicting information

Respond with ONLY 'correct' or 'incorrect'.
Response:"""
        
        response = self.llm_model.infer([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        if response.strip().lower() == "correct":
            return 1.0
        else:
            return 0.0
    
    def _evaluate_single(self, idx: int, result: Dict[str, Any], compute_llm: bool) -> tuple:
        """Evaluate a single result.
        
        Args:
            idx: Result index
            result: Result dict with pred_answer and gold_answer
            compute_llm: Whether to compute LLM accuracy
            
        Returns:
            Tuple of (idx, contain_score, llm_score)
        """
        pred_answer = result.get("pred_answer", "")
        gold_answer = result.get("gold_answer", "")
        
        contain_score = self.calculate_contain(pred_answer, gold_answer)
        llm_score = 0.0
        
        if compute_llm and self.llm_model is not None:
            llm_score = self.calculate_llm_accuracy(pred_answer, gold_answer)
        
        return idx, contain_score, llm_score
    
    def evaluate(
        self, 
        compute_llm: bool = False, 
        max_workers: int = 2,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Evaluate all results.
        
        Args:
            compute_llm: Whether to compute LLM accuracy (requires llm_model)
            max_workers: Number of parallel workers for LLM evaluation
            show_progress: Whether to show progress bar
            
        Returns:
            Dict with metrics:
                - contain_accuracy: Fraction of predictions containing gold answer
                - llm_accuracy: Fraction judged correct by LLM (if compute_llm=True)
                - contain_scores: Per-sample contain scores
                - llm_scores: Per-sample LLM scores (if compute_llm=True)
        """
        n = len(self.results)
        contain_scores = [0] * n
        llm_scores = [0.0] * n
        
        if compute_llm and self.llm_model is not None:
            # Parallel evaluation with LLM
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._evaluate_single, idx, result, True): idx
                    for idx, result in enumerate(self.results)
                }
                
                iterator = as_completed(futures)
                if show_progress:
                    iterator = tqdm(iterator, total=len(futures), desc="Evaluating answers")
                
                for future in iterator:
                    idx, contain, llm = future.result()
                    contain_scores[idx] = contain
                    llm_scores[idx] = llm
        else:
            # Fast evaluation without LLM
            iterator = enumerate(self.results)
            if show_progress:
                iterator = tqdm(iterator, total=n, desc="Evaluating (contain only)")
            
            for idx, result in iterator:
                contain_scores[idx] = self.calculate_contain(
                    result.get("pred_answer", ""),
                    result.get("gold_answer", "")
                )
        
        contain_accuracy = sum(contain_scores) / n if n > 0 else 0.0
        llm_accuracy = sum(llm_scores) / n if n > 0 and compute_llm else 0.0
        
        metrics = {
            "contain_accuracy": contain_accuracy,
            "contain_scores": contain_scores,
            "total_samples": n
        }
        
        if compute_llm:
            metrics["llm_accuracy"] = llm_accuracy
            metrics["llm_scores"] = llm_scores
        
        return metrics

