"""Utility functions for LinearRAG including hashing, normalization, and logging."""

from hashlib import md5
import re
import string
import logging
import os
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute MD5 hash ID for content with optional prefix.
    
    Args:
        content: Text content to hash
        prefix: Optional prefix for the hash ID
        
    Returns:
        Hash ID string with prefix
    """
    return prefix + md5(content.encode()).hexdigest()


def normalize_answer(s: Union[str, None]) -> str:
    """Normalize answer text for comparison.
    
    Removes articles, punctuation, extra whitespace, and lowercases text.
    
    Args:
        s: Text to normalize
        
    Returns:
        Normalized text string
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())
    
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text: str) -> str:
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1] range.
    
    Args:
        x: Input numpy array
        
    Returns:
        Normalized array with values in [0, 1]
    """
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    if range_val == 0:
        return np.ones_like(x)
    
    return (x - min_val) / range_val


def strip_passage_prefix(passage: str) -> str:
    """Remove index prefix like '0:' from passage.
    
    Args:
        passage: Passage text potentially with index prefix
        
    Returns:
        Passage text without index prefix
    """
    if ":" in passage and passage.split(":")[0].isdigit():
        return passage.split(":", 1)[1]
    return passage


def setup_logging(log_file: str) -> None:
    """Setup logging configuration with file and console handlers.
    
    Args:
        log_file: Path to log file
    """
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

