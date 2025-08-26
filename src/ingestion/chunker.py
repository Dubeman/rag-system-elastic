"""Text chunking utilities for document processing."""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


class TextChunker:
    """Text chunking with configurable strategies."""

    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // 4

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', ' ', text)
        return text.strip()

    def chunk_text(self, text: str) -> List[Dict]:
        """Chunk text into smaller pieces with metadata."""
        if not text or not text.strip():
            return []

        cleaned_text = self.clean_text(text)
        words = cleaned_text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            word_tokens = self.estimate_tokens(word)
            
            if current_size + word_tokens > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "chunk_id": len(chunks),
                    "text": chunk_text,
                    "token_count": current_size,
                    "char_count": len(chunk_text),
                })
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    current_chunk = current_chunk[-self.chunk_overlap:]
                    current_size = sum(self.estimate_tokens(w) for w in current_chunk)
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(word)
            current_size += word_tokens

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "chunk_id": len(chunks),
                "text": chunk_text,
                "token_count": current_size,
                "char_count": len(chunk_text),
            })

        logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")
        return chunks

    def chunk_document(self, document: Dict) -> Dict:
        """Chunk a complete document with metadata preservation."""
        text = document.get("text", "")
        if not text:
            return {**document, "chunks": []}

        chunks = self.chunk_text(text)
        
        # Add document metadata to each chunk
        for chunk in chunks:
            chunk.update({
                "document_id": document.get("file_id", ""),
                "filename": document.get("filename", ""),
                "source_url": document.get("file_url", ""),
            })

        return {**document, "chunks": chunks, "chunk_count": len(chunks)}
