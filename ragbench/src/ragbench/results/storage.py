import json
from pathlib import Path
from typing import List, Dict

from .models import ExperimentResult


def save_results(results: List[ExperimentResult], output_path: str):
    """
    Save experiment results to JSON file.
    
    Args:
        results: List of experiment results
        output_path: Path to output JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = [result.to_dict() for result in results]
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_results(input_path: str) -> List[ExperimentResult]:
    """
    Load experiment results from JSON file.
    
    Args:
        input_path: Path to input JSON file
    
    Returns:
        List of experiment results
    """
    with open(input_path, "r") as f:
        data = json.load(f)
    
    return [ExperimentResult.from_dict(item) for item in data]


def save_result(result: ExperimentResult, output_path: str, append: bool = True):
    """
    Save a single experiment result to JSONL file (one JSON object per line).
    
    Args:
        result: Single experiment result
        output_path: Path to output JSONL file
        append: If True, append to existing file; if False, overwrite
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mode = "a" if append and output_path.exists() else "w"
    with open(output_path, mode) as f:
        json.dump(result.to_dict(), f)
        f.write("\n")


def save_results_csv(results: List[ExperimentResult], output_path: str):
    """
    Save experiment results to CSV file.
    
    Args:
        results: List of experiment results
        output_path: Path to output CSV file
    """
    import csv
    
    if not results:
        return
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Flatten config and metrics into single row
    fieldnames = ["timestamp"]
    
    # Get all unique config keys
    config_keys = set()
    metric_keys = set()
    for result in results:
        config_keys.update(result.config.keys())
        metric_keys.update(result.metrics.keys())
    
    fieldnames.extend(sorted(config_keys))
    fieldnames.extend(sorted(metric_keys))
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {"timestamp": result.timestamp}
            row.update(result.config)
            row.update(result.metrics)
            writer.writerow(row)
