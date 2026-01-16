"""Parquet-based embedding storage for LinearRAG.

Stores text embeddings with hash-based deduplication and fast lookup.
Also stores document metadata (file_source, page_num) required for evaluation.
"""

from copy import deepcopy
from typing import List, Dict, Any, Union, Optional
import os
import logging

import numpy as np
import pandas as pd

from ..utils import compute_mdhash_id

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Parquet-based storage for text embeddings with hash-based deduplication.
    
    Stores texts, their embeddings, and provides fast lookup by hash ID.
    Used for passages, entities, and sentences in LinearRAG.
    
    Also stores document metadata (file_source, page_num) for each entry.
    This metadata is required for evaluation - without it, the evaluator cannot
    compare retrieved passages against ground truth documents.
    
    Args:
        embedding_model: Model implementing EmbeddingModelProtocol for encoding texts
        db_filename: Path to parquet file for persistence
        batch_size: Batch size for encoding
        namespace: Prefix for hash IDs (e.g., 'passage', 'entity', 'sentence')
    """
    
    def __init__(
        self, 
        embedding_model: Any,
        db_filename: str,
        batch_size: int = 128,
        namespace: str = "default"
    ):
        self.embedding_model = embedding_model
        self.db_filename = db_filename
        self.batch_size = batch_size
        self.namespace = namespace
        
        # In-memory storage
        self.hash_ids: List[str] = []
        self.texts: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []  # Stores file_source, page_num
        
        # Lookup dictionaries
        self.hash_id_to_text: Dict[str, str] = {}
        self.hash_id_to_idx: Dict[str, int] = {}
        self.text_to_hash_id: Dict[str, str] = {}
        self.hash_id_to_metadata: Dict[str, Dict[str, Any]] = {}
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load existing data from parquet file if it exists."""
        if os.path.exists(self.db_filename):
            try:
                df = pd.read_parquet(self.db_filename)
                self.hash_ids = df["hash_id"].values.tolist()
                self.texts = df["text"].values.tolist()
                self.embeddings = df["embedding"].values.tolist()
                
                # Load metadata if present (backwards compatible)
                if "file_source" in df.columns and "page_num" in df.columns:
                    self.metadata = [
                        {"file_source": fs, "page_num": pn}
                        for fs, pn in zip(df["file_source"].values, df["page_num"].values)
                    ]
                else:
                    # No metadata in old files - use empty dicts
                    self.metadata = [{} for _ in self.hash_ids]
                
                self._rebuild_indices()
                logger.info(f"[{self.namespace}] Loaded {len(self.hash_ids)} records from {self.db_filename}")
            except Exception as e:
                logger.warning(f"[{self.namespace}] Failed to load data: {e}")
    
    def _rebuild_indices(self) -> None:
        """Rebuild lookup dictionaries from lists."""
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts)}
        self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids)}
        self.hash_id_to_metadata = {h: m for h, m in zip(self.hash_ids, self.metadata)}
    
    def insert_text(
        self, 
        text_list: Union[List[str], List[Dict[str, Any]]]
    ) -> None:
        """Insert texts and compute their embeddings.
        
        Automatically deduplicates based on content hash.
        
        Args:
            text_list: Either a list of text strings, or a list of dicts with keys:
                - "text": The passage text (required)
                - "file_source": Source document filename (required for evaluation)
                - "page_num": Page number in source document (required for evaluation)
                
        Note:
            Plain text strings are supported for backwards compatibility, but
            metadata (file_source, page_num) is required if you want to use
            the evaluation framework.
        """
        if not text_list:
            return
        
        # Normalize input to dicts
        if isinstance(text_list[0], str):
            items = [{"text": t, "file_source": "", "page_num": 0} for t in text_list]
        else:
            items = [
                {
                    "text": item["text"],
                    "file_source": item.get("file_source", ""),
                    "page_num": item.get("page_num", 0)
                }
                for item in text_list
            ]
            
        # Build hash-to-content mapping
        nodes_dict = {}
        for item in items:
            hash_id = compute_mdhash_id(item["text"], prefix=self.namespace + "-")
            nodes_dict[hash_id] = {
                'content': item["text"],
                'metadata': {
                    "file_source": item["file_source"],
                    "page_num": item["page_num"]
                }
            }
        
        all_hash_ids = list(nodes_dict.keys())
        
        # Find texts that don't exist yet
        existing = set(self.hash_ids)
        missing_ids = [h for h in all_hash_ids if h not in existing]
        
        if not missing_ids:
            return
            
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        metadata_to_store = [nodes_dict[hash_id]["metadata"] for hash_id in missing_ids]
        
        # Encode new texts
        all_embeddings = self.embedding_model.encode(
            texts_to_encode,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.batch_size
        )
        
        self._upsert(missing_ids, texts_to_encode, all_embeddings, metadata_to_store)
    
    def _upsert(
        self, 
        hash_ids: List[str], 
        texts: List[str], 
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add new entries to storage and save.
        
        Args:
            hash_ids: List of hash IDs for new entries
            texts: List of texts
            embeddings: Embedding vectors as numpy array
            metadata: Optional list of metadata dicts with file_source and page_num
        """
        # Convert embeddings to list if needed
        if isinstance(embeddings, np.ndarray):
            embeddings_list = [emb for emb in embeddings]
        else:
            embeddings_list = list(embeddings)
        
        # Use empty metadata if not provided
        if metadata is None:
            metadata = [{} for _ in hash_ids]
        
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        self.embeddings.extend(embeddings_list)
        self.metadata.extend(metadata)
        
        self._rebuild_indices()
        self._save_data()
    
    def _save_data(self) -> None:
        """Save data to parquet file including metadata."""
        # Extract file_source and page_num from metadata dicts
        file_sources = [m.get("file_source", "") for m in self.metadata]
        page_nums = [m.get("page_num", 0) for m in self.metadata]
        
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "text": self.texts,
            "embedding": self.embeddings,
            "file_source": file_sources,
            "page_num": page_nums
        })
        
        os.makedirs(os.path.dirname(self.db_filename), exist_ok=True)
        data_to_save.to_parquet(self.db_filename, index=False)
    
    def get_hash_id_to_text(self) -> Dict[str, str]:
        """Get a copy of hash_id to text mapping.
        
        Returns:
            Dictionary mapping hash IDs to text content
        """
        return deepcopy(self.hash_id_to_text)
    
    def get_hash_id_to_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get a copy of hash_id to metadata mapping.
        
        Returns:
            Dictionary mapping hash IDs to metadata dicts with file_source and page_num
        """
        return deepcopy(self.hash_id_to_metadata)
    
    def get_metadata(self, hash_id: str) -> Dict[str, Any]:
        """Get metadata for a specific hash ID.
        
        Args:
            hash_id: The hash ID to look up
            
        Returns:
            Metadata dict with file_source and page_num, or empty dict if not found
        """
        return self.hash_id_to_metadata.get(hash_id, {})
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts using the embedding model.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Numpy array of embeddings
        """
        return self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.batch_size
        )
    
    def get_embeddings(self, hash_ids: List[str]) -> np.ndarray:
        """Get embeddings for given hash IDs.
        
        Args:
            hash_ids: List of hash IDs to retrieve
            
        Returns:
            Numpy array of embeddings
        """
        if not hash_ids:
            return np.array([])
        
        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings)[indices]
        return embeddings
    
    def __len__(self) -> int:
        """Return number of stored entries."""
        return len(self.hash_ids)

