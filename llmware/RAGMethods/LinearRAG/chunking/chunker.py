"""Chunking strategies for LinearRAG.

Three chunking strategies:
1. Original (no rechunking) - use chunks as-is (may be truncated by embedding model)
2. Fixed-size + Overlap - split into fixed token chunks with overlap
3. Sentence-based - group sentences into semantic chunks

All strategies use the actual tokenizer from sentence-transformers for accurate token counting.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Union, Any

from sentence_transformers import SentenceTransformer

from .config import ChunkingConfig

logger = logging.getLogger(__name__)


class Tokenizer:
    """Wrapper for sentence-transformers tokenizer.
    
    Uses singleton pattern to avoid reloading model for each tokenization call.
    The tokenizer is loaded lazily on first use.
    """
    
    _tokenizer = None
    _max_seq_length = None
    _model_name = None
    
    @classmethod
    def get_tokenizer(cls, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """Get or create tokenizer instance (singleton).
        
        Args:
            model_name: Name of the sentence-transformers model to use for tokenization.
            
        Returns:
            The tokenizer from the specified model.
        """
        if cls._tokenizer is None or cls._model_name != model_name:
            logger.info(f"Loading tokenizer from {model_name}...")
            model = SentenceTransformer(model_name)
            cls._tokenizer = model.tokenizer
            cls._max_seq_length = model.max_seq_length
            cls._model_name = model_name
            logger.info(f"Tokenizer loaded. Max seq length: {cls._max_seq_length}")
        return cls._tokenizer
    
    @classmethod
    def get_max_seq_length(cls, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> int:
        """Get max sequence length for the model.
        
        Args:
            model_name: Name of the sentence-transformers model.
            
        Returns:
            Maximum sequence length supported by the model.
        """
        if cls._max_seq_length is None or cls._model_name != model_name:
            cls.get_tokenizer(model_name)
        return cls._max_seq_length
    
    @classmethod
    def count_tokens(cls, text: str, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> int:
        """Count tokens in text using the actual tokenizer.
        
        Args:
            text: Text to tokenize.
            model_name: Name of the sentence-transformers model.
            
        Returns:
            Number of tokens in the text.
        """
        tokenizer = cls.get_tokenizer(model_name)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    
    @classmethod
    def tokenize(cls, text: str, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> List[int]:
        """Tokenize text and return token IDs.
        
        Args:
            text: Text to tokenize.
            model_name: Name of the sentence-transformers model.
            
        Returns:
            List of token IDs.
        """
        tokenizer = cls.get_tokenizer(model_name)
        return tokenizer.encode(text, add_special_tokens=False)
    
    @classmethod
    def decode(cls, token_ids: List[int], model_name: str = "sentence-transformers/all-mpnet-base-v2") -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode.
            model_name: Name of the sentence-transformers model.
            
        Returns:
            Decoded text string.
        """
        tokenizer = cls.get_tokenizer(model_name)
        return tokenizer.decode(token_ids, skip_special_tokens=True)


class ChunkingStrategy:
    """Base class for chunking strategies.
    
    Subclasses must implement the `chunk` method to define their chunking behavior.
    
    Args:
        config: ChunkingConfig with strategy parameters.
    """
    
    name: str = "base"
    description: str = "Base chunking strategy"
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.model_name = self.config.embedding_model_name
    
    def chunk(
        self, 
        texts: Union[List[str], List[Dict[str, Any]]]
    ) -> Tuple[List[Union[str, Dict[str, Any]]], Dict[str, Any]]:
        """Chunk texts and return (chunks, metadata).
        
        Args:
            texts: Either a list of text strings, or a list of dicts with:
                - "text": The passage text (required)
                - "file_source": Source document filename (optional)
                - "page_num": Page number in source document (optional)
                
        Returns:
            Tuple of (chunked_texts, chunking_metadata) where:
                - chunked_texts maintains the same format as input (str or dict)
                - chunking_metadata contains statistics about the chunking process
        """
        raise NotImplementedError("Subclasses must implement chunk()")
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Remove index prefix if present (e.g., '0:text' -> 'text').
        
        Args:
            text: Text that may have an index prefix.
            
        Returns:
            Text with index prefix removed.
        """
        return re.sub(r'^\d+:', '', text).strip()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using actual tokenizer.
        
        Args:
            text: Text to count tokens for.
            
        Returns:
            Number of tokens.
        """
        return Tokenizer.count_tokens(text, self.model_name)
    
    def _extract_text(self, item: Union[str, Dict[str, Any]]) -> str:
        """Extract text content from input item.
        
        Args:
            item: Either a string or dict with "text" key.
            
        Returns:
            The text content.
        """
        if isinstance(item, str):
            return item
        return item.get("text", "")
    
    def _extract_metadata(self, item: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metadata from input item.
        
        Args:
            item: Either a string or dict with metadata.
            
        Returns:
            Metadata dict with file_source and page_num.
        """
        if isinstance(item, str):
            return {"file_source": "", "page_num": 0}
        return {
            "file_source": item.get("file_source", ""),
            "page_num": item.get("page_num", 0)
        }
    
    def _create_output_item(
        self, 
        text: str, 
        metadata: Dict[str, Any],
        is_dict_input: bool
    ) -> Union[str, Dict[str, Any]]:
        """Create output item matching input format.
        
        Args:
            text: The chunk text.
            metadata: Metadata to include if dict format.
            is_dict_input: Whether input was dict format.
            
        Returns:
            String or dict matching input format.
        """
        if is_dict_input:
            return {
                "text": text,
                "file_source": metadata.get("file_source", ""),
                "page_num": metadata.get("page_num", 0)
            }
        return text


class OriginalChunking(ChunkingStrategy):
    """Strategy 1: Original Chunks (No Rechunking).
    
    Uses chunks as-is. Chunks exceeding the embedding model's max sequence length
    will be TRUNCATED during embedding. This provides a baseline for comparison.
    """
    
    name = "original"
    description = "Original chunks (may be truncated during embedding)"
    
    def chunk(
        self, 
        texts: Union[List[str], List[Dict[str, Any]]]
    ) -> Tuple[List[Union[str, Dict[str, Any]]], Dict[str, Any]]:
        """Return original texts with metadata about truncation.
        
        Args:
            texts: Input texts (strings or dicts with text/metadata).
            
        Returns:
            Tuple of (original_texts, chunking_metadata).
        """
        if not texts:
            return [], {"strategy": self.name, "error": "No input texts"}
        
        is_dict_input = isinstance(texts[0], dict)
        chunks = []
        
        for item in texts:
            text = self._extract_text(item)
            metadata = self._extract_metadata(item)
            clean = self.clean_text(text)
            chunks.append(self._create_output_item(clean, metadata, is_dict_input))
        
        # Compute statistics using real tokenizer
        logger.info("Computing token statistics with real tokenizer...")
        chunk_texts = [self._extract_text(c) for c in chunks]
        token_counts = [self.count_tokens(c) for c in chunk_texts]
        
        max_seq_len = Tokenizer.get_max_seq_length(self.model_name)
        over_limit = sum(1 for t in token_counts if t > max_seq_len)
        
        metadata = {
            "strategy": self.name,
            "description": self.description,
            "model_max_seq_length": max_seq_len,
            "original_count": len(texts),
            "output_count": len(chunks),
            "expansion_ratio": 1.0,
            "statistics": {
                "min_tokens": min(token_counts) if token_counts else 0,
                "max_tokens": max(token_counts) if token_counts else 0,
                "mean_tokens": round(sum(token_counts) / len(token_counts)) if token_counts else 0,
                "median_tokens": sorted(token_counts)[len(token_counts)//2] if token_counts else 0,
                "chunks_over_limit": over_limit,
                "truncation_rate": round(over_limit / len(chunks) * 100, 1) if chunks else 0
            },
            "warning": f"{over_limit}/{len(chunks)} chunks ({over_limit/len(chunks)*100:.1f}%) will be TRUNCATED" if chunks else ""
        }
        
        logger.info(f"[{self.name}] {len(chunks)} chunks, {over_limit} will be truncated (>{max_seq_len} tokens)")
        logger.info(f"[{self.name}] Token stats: min={metadata['statistics']['min_tokens']}, "
                   f"max={metadata['statistics']['max_tokens']}, mean={metadata['statistics']['mean_tokens']}")
        
        return chunks, metadata


class FixedSizeChunking(ChunkingStrategy):
    """Strategy 2: Fixed-Size Chunking with Overlap (Token-based).
    
    Splits text into fixed-size TOKEN chunks with overlap.
    - Default: 300 tokens with 50 token overlap
    - Uses actual tokenizer for precise splitting
    - Ensures all chunks fit within embedding model's token limit
    """
    
    name = "fixed_size"
    description = "Fixed-size token chunks with overlap"
    
    def chunk(
        self, 
        texts: Union[List[str], List[Dict[str, Any]]]
    ) -> Tuple[List[Union[str, Dict[str, Any]]], Dict[str, Any]]:
        """Split texts into fixed-size overlapping chunks using tokenizer.
        
        Args:
            texts: Input texts (strings or dicts with text/metadata).
            
        Returns:
            Tuple of (chunked_texts, chunking_metadata).
        """
        if not texts:
            return [], {"strategy": self.name, "error": "No input texts"}
        
        max_tokens = self.config.fixed_max_tokens
        overlap = self.config.fixed_overlap_tokens
        min_tokens = self.config.min_chunk_tokens
        
        is_dict_input = isinstance(texts[0], dict)
        all_chunks = []
        
        logger.info(f"[{self.name}] Chunking with max_tokens={max_tokens}, overlap={overlap}")
        
        for item in texts:
            text = self._extract_text(item)
            item_metadata = self._extract_metadata(item)
            clean = self.clean_text(text)
            
            # Tokenize the full text
            token_ids = Tokenizer.tokenize(clean, self.model_name)
            
            if len(token_ids) <= max_tokens:
                # Short enough, keep as-is
                if len(token_ids) >= min_tokens:
                    all_chunks.append(self._create_output_item(clean, item_metadata, is_dict_input))
            else:
                # Split into overlapping chunks
                start = 0
                while start < len(token_ids):
                    end = start + max_tokens
                    chunk_tokens = token_ids[start:end]
                    
                    # Decode back to text
                    chunk_text = Tokenizer.decode(chunk_tokens, self.model_name)
                    
                    if len(chunk_tokens) >= min_tokens:
                        all_chunks.append(self._create_output_item(
                            chunk_text.strip(), item_metadata, is_dict_input
                        ))
                    
                    # Move forward with overlap
                    start += (max_tokens - overlap)
                    if start >= len(token_ids):
                        break
        
        # Compute statistics
        chunk_texts = [self._extract_text(c) for c in all_chunks]
        token_counts = [self.count_tokens(c) for c in chunk_texts]
        max_seq_len = Tokenizer.get_max_seq_length(self.model_name)
        over_limit = sum(1 for t in token_counts if t > max_seq_len)
        
        metadata = {
            "strategy": self.name,
            "description": f"{self.description} ({max_tokens} tokens, {overlap} overlap)",
            "model_max_seq_length": max_seq_len,
            "parameters": {
                "max_tokens": max_tokens,
                "overlap_tokens": overlap,
                "min_chunk_tokens": min_tokens
            },
            "original_count": len(texts),
            "output_count": len(all_chunks),
            "expansion_ratio": round(len(all_chunks) / len(texts), 2) if texts else 0,
            "statistics": {
                "min_tokens": min(token_counts) if token_counts else 0,
                "max_tokens": max(token_counts) if token_counts else 0,
                "mean_tokens": round(sum(token_counts) / len(token_counts)) if token_counts else 0,
                "chunks_over_limit": over_limit
            }
        }
        
        logger.info(f"[{self.name}] {len(texts)} -> {len(all_chunks)} chunks "
                   f"(expansion: {metadata['expansion_ratio']}x)")
        logger.info(f"[{self.name}] Token stats: min={metadata['statistics']['min_tokens']}, "
                   f"max={metadata['statistics']['max_tokens']}, mean={metadata['statistics']['mean_tokens']}")
        
        return all_chunks, metadata


class SentenceBasedChunking(ChunkingStrategy):
    """Strategy 3: Sentence-Based Chunking.
    
    Groups sentences into semantically coherent chunks.
    - Default: 4 sentences per chunk with 1 sentence overlap
    - Respects sentence boundaries (no mid-sentence cuts)
    - Validates token count to stay under limit
    """
    
    name = "sentence_based"
    description = "Sentence-based chunks with overlap"
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into sentences using regex.
        
        Handles common abbreviations and edge cases.
        
        Args:
            text: Text to split into sentences.
            
        Returns:
            List of sentences.
        """
        # Add space after sentence-ending punctuation if not present
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Split on sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*(?=")|(?<=[.!?"])(?=\s+[A-Z])'
        
        sentences = re.split(sentence_pattern, text)
        
        # Clean up and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If regex didn't work well, fall back to simple split
        if len(sentences) <= 1 and len(text) > 200:
            sentences = [s.strip() + '.' for s in text.split('. ') if s.strip()]
        
        return sentences
    
    def chunk(
        self, 
        texts: Union[List[str], List[Dict[str, Any]]]
    ) -> Tuple[List[Union[str, Dict[str, Any]]], Dict[str, Any]]:
        """Group sentences into chunks, respecting token limits.
        
        Args:
            texts: Input texts (strings or dicts with text/metadata).
            
        Returns:
            Tuple of (chunked_texts, chunking_metadata).
        """
        if not texts:
            return [], {"strategy": self.name, "error": "No input texts"}
        
        sentences_per_chunk = self.config.sentences_per_chunk
        overlap = self.config.sentence_overlap
        max_tokens = self.config.max_chunk_tokens
        min_tokens = self.config.min_chunk_tokens
        
        is_dict_input = isinstance(texts[0], dict)
        all_chunks = []
        
        logger.info(f"[{self.name}] Chunking with sentences_per_chunk={sentences_per_chunk}, "
                   f"overlap={overlap}, max_tokens={max_tokens}")
        
        for item in texts:
            text = self._extract_text(item)
            item_metadata = self._extract_metadata(item)
            clean = self.clean_text(text)
            sentences = self.split_sentences(clean)
            
            if len(sentences) <= sentences_per_chunk:
                # Few sentences, keep as one chunk
                chunk = ' '.join(sentences)
                if self.count_tokens(chunk) >= min_tokens:
                    all_chunks.append(self._create_output_item(chunk, item_metadata, is_dict_input))
            else:
                # Group sentences with overlap
                start = 0
                while start < len(sentences):
                    end = start + sentences_per_chunk
                    
                    # Get candidate chunk
                    chunk_sentences = sentences[start:end]
                    chunk = ' '.join(chunk_sentences)
                    
                    # Adjust if over token limit
                    while self.count_tokens(chunk) > max_tokens and len(chunk_sentences) > 1:
                        chunk_sentences = chunk_sentences[:-1]
                        chunk = ' '.join(chunk_sentences)
                    
                    # Add more sentences if under minimum and available
                    while self.count_tokens(chunk) < 50 and end < len(sentences):
                        end += 1
                        chunk_sentences = sentences[start:end]
                        chunk = ' '.join(chunk_sentences)
                        if self.count_tokens(chunk) > max_tokens:
                            # Went over, remove last
                            chunk_sentences = chunk_sentences[:-1]
                            chunk = ' '.join(chunk_sentences)
                            break
                    
                    if self.count_tokens(chunk) >= min_tokens:
                        all_chunks.append(self._create_output_item(chunk, item_metadata, is_dict_input))
                    
                    # Move forward with overlap
                    actual_sentences_used = len(chunk_sentences)
                    start += max(1, actual_sentences_used - overlap)
                    
                    if start >= len(sentences):
                        break
        
        # Compute statistics
        chunk_texts = [self._extract_text(c) for c in all_chunks]
        token_counts = [self.count_tokens(c) for c in chunk_texts]
        max_seq_len = Tokenizer.get_max_seq_length(self.model_name)
        over_limit = sum(1 for t in token_counts if t > max_seq_len)
        
        metadata = {
            "strategy": self.name,
            "description": f"{self.description} ({sentences_per_chunk} sentences, {overlap} overlap)",
            "model_max_seq_length": max_seq_len,
            "parameters": {
                "sentences_per_chunk": sentences_per_chunk,
                "sentence_overlap": overlap,
                "max_chunk_tokens": max_tokens,
                "min_chunk_tokens": min_tokens
            },
            "original_count": len(texts),
            "output_count": len(all_chunks),
            "expansion_ratio": round(len(all_chunks) / len(texts), 2) if texts else 0,
            "statistics": {
                "min_tokens": min(token_counts) if token_counts else 0,
                "max_tokens": max(token_counts) if token_counts else 0,
                "mean_tokens": round(sum(token_counts) / len(token_counts)) if token_counts else 0,
                "chunks_over_limit": over_limit
            }
        }
        
        logger.info(f"[{self.name}] {len(texts)} -> {len(all_chunks)} chunks "
                   f"(expansion: {metadata['expansion_ratio']}x)")
        logger.info(f"[{self.name}] Token stats: min={metadata['statistics']['min_tokens']}, "
                   f"max={metadata['statistics']['max_tokens']}, mean={metadata['statistics']['mean_tokens']}")
        
        return all_chunks, metadata


def get_chunking_strategy(
    strategy_name: str, 
    config: Optional[ChunkingConfig] = None
) -> ChunkingStrategy:
    """Get chunking strategy by name.
    
    Factory function to create the appropriate chunking strategy instance.
    
    Args:
        strategy_name: One of 'original', 'fixed_size', 'sentence_based'.
        config: Optional ChunkingConfig with strategy parameters.
        
    Returns:
        ChunkingStrategy instance for the specified strategy.
        
    Raises:
        ValueError: If strategy_name is not recognized.
    """
    strategies = {
        "original": OriginalChunking,
        "fixed_size": FixedSizeChunking,
        "sentence_based": SentenceBasedChunking
    }
    
    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Choose from {list(strategies.keys())}"
        )
    
    return strategies[strategy_name](config)

