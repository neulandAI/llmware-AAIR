from typing import List, Dict

from .base import Chunker


class FixedSizeChunker(Chunker):
    """Fixed-size chunker with optional overlap."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize fixed-size chunker.
        
        Args:
            chunk_size: Size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, documents: List[Dict]) -> List[Dict]:
        """Chunk documents into fixed-size pieces."""
        chunks = []
        
        for doc in documents:
            text = doc.get("text", "")
            doc_id = doc.get("doc_id", doc.get("id", ""))
            
            if not text:
                continue
            
            # Split text into chunks
            start = 0
            chunk_idx = 0
            
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                
                chunk = {
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                    "metadata": {
                        "start": start,
                        "end": end,
                        "chunk_size": len(chunk_text),
                    },
                }
                
                # Copy any additional metadata from document
                if "metadata" in doc:
                    chunk["metadata"].update(doc["metadata"])
                
                chunks.append(chunk)
                
                # Move start position with overlap
                start = end - self.overlap
                chunk_idx += 1
        
        return chunks
