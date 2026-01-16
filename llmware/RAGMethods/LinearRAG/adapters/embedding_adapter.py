"""Embedding model adapters for LinearRAG.

Provides a protocol for embedding models and implementations for
SentenceTransformer and llmware embedding models.
"""

from typing import List, Union, Protocol, runtime_checkable
import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingModelProtocol(Protocol):
    """Protocol for embedding models used by LinearRAG.
    
    Any embedding model must implement this interface to work with LinearRAG.
    The key method is `encode` which takes texts and returns embeddings.
    """
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32
    ) -> np.ndarray:
        """Encode texts into embedding vectors.
        
        Args:
            texts: Single text string or list of texts
            normalize_embeddings: Whether to L2-normalize embeddings
            show_progress_bar: Whether to show progress during encoding
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        ...


class SentenceTransformerAdapter:
    """Adapter for sentence-transformers models.
    
    Wraps a SentenceTransformer model to implement EmbeddingModelProtocol.
    This is the recommended adapter for LinearRAG as it provides efficient
    batch encoding.
    
    Args:
        model_name_or_path: Name or path of SentenceTransformer model
        device: Device to use ('cuda', 'cpu', or None for auto)
        
    Example:
        >>> adapter = SentenceTransformerAdapter("all-mpnet-base-v2")
        >>> embeddings = adapter.encode(["Hello world", "Test text"])
    """
    
    def __init__(self, model_name_or_path: str, device: str = None):
        self.model_name = model_name_or_path
        self.model = SentenceTransformer(model_name_or_path, device=device)
        logger.info(f"Loaded SentenceTransformer model: {model_name_or_path}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32
    ) -> np.ndarray:
        """Encode texts using SentenceTransformer.
        
        Args:
            texts: Single text or list of texts
            normalize_embeddings: Whether to L2-normalize
            show_progress_bar: Show progress bar
            batch_size: Batch size
            
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size
        )


class LLMWareEmbeddingAdapter:
    """Adapter for llmware embedding models.
    
    Wraps an llmware embedding model to implement EmbeddingModelProtocol.
    Converts llmware's `.embedding()` interface to the batch `.encode()` interface.
    
    Args:
        llmware_model: An llmware embedding model with `.embedding()` method
        
    Example:
        >>> from llmware.models import ModelCatalog
        >>> llmware_emb = ModelCatalog().load_model("industry-bert-sec")
        >>> adapter = LLMWareEmbeddingAdapter(llmware_emb)
        >>> embeddings = adapter.encode(["Hello world", "Test text"])
    """
    
    def __init__(self, llmware_model):
        self.model = llmware_model
        
        # Check that model has embedding method
        if not hasattr(llmware_model, 'embedding'):
            raise ValueError(
                "llmware model must have an 'embedding' method. "
                "Make sure you're using an embedding model, not a generative model."
            )
        
        logger.info(f"Wrapped llmware embedding model: {getattr(llmware_model, 'model_name', 'unknown')}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """Encode texts using llmware embedding model.
        
        Args:
            texts: Single text or list of texts
            normalize_embeddings: Whether to L2-normalize
            
        Returns:
            Numpy array of embeddings
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Encode all texts
        embeddings = []
        for text in texts:
            emb = self.model.embedding(text)
            # Handle different return formats from llmware
            if isinstance(emb, list):
                emb = np.array(emb)
            if len(emb.shape) == 2:
                emb = emb[0]  # Take first if batched
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # Normalize if requested
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            embeddings = embeddings / norms
        
        return embeddings

