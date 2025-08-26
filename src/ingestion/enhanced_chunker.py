"""Enhanced text chunking with LangChain integration."""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import LangChain, fallback to custom implementation
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain text splitters available")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available, using custom chunking")

from .chunker import TextChunker


class EnhancedChunker:
    """Enhanced text chunker with LangChain integration and fallback."""

    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        use_langchain: bool = True,
    ):
        """
        Initialize enhanced chunker.

        Args:
            chunk_size: Target chunk size in characters/tokens
            chunk_overlap: Overlap between chunks
            use_langchain: Whether to use LangChain splitter if available
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE

        if self.use_langchain:
            self._init_langchain_splitter()
        else:
            self._init_custom_splitter()

    def _init_langchain_splitter(self) -> None:
        """Initialize LangChain RecursiveCharacterTextSplitter."""
        try:
            self.langchain_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n",  # Double newlines (paragraphs)
                    "\n",    # Single newlines
                    ". ",    # Sentences
                    "! ",    # Exclamations
                    "? ",    # Questions
                    "; ",    # Semicolons
                    ", ",    # Commas
                    " ",     # Spaces
                    "",      # Characters
                ],
                keep_separator=False,
            )
            logger.info("LangChain RecursiveCharacterTextSplitter initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain splitter: {e}")
            self.use_langchain = False
            self._init_custom_splitter()

    def _init_custom_splitter(self) -> None:
        """Initialize custom text chunker as fallback."""
        self.custom_chunker = TextChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        logger.info("Custom text chunker initialized")

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // 4

    def chunk_text_langchain(self, text: str) -> List[str]:
        """Chunk text using LangChain splitter."""
        try:
            chunks = self.langchain_splitter.split_text(text)
            logger.debug(f"LangChain created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"LangChain chunking failed: {e}")
            # Fallback to custom chunker
            return self.chunk_text_custom(text)

    def chunk_text_custom(self, text: str) -> List[str]:
        """Chunk text using custom chunker."""
        chunk_dicts = self.custom_chunker.chunk_text(text)
        return [chunk["text"] for chunk in chunk_dicts]

    def chunk_text(self, text: str) -> List[Dict]:
        """
        Chunk text into smaller pieces with metadata.

        Args:
            text: Input text to chunk

        Returns:
            List of chunk dictionaries with metadata
        """
        if not text or not text.strip():
            return []

        # Choose chunking method
        if self.use_langchain:
            text_chunks = self.chunk_text_langchain(text)
            method = "langchain"
        else:
            text_chunks = self.chunk_text_custom(text)
            method = "custom"

        # Convert to standardized format with metadata
        chunk_objects = []
        for i, chunk_text in enumerate(text_chunks):
            if chunk_text.strip():  # Only include non-empty chunks
                chunk_objects.append({
                    "chunk_id": i,
                    "text": chunk_text.strip(),
                    "token_count": self.estimate_tokens(chunk_text),
                    "char_count": len(chunk_text),
                    "chunking_method": method,
                })

        logger.info(f"Created {len(chunk_objects)} chunks using {method} method")
        return chunk_objects

    def chunk_document(self, document: Dict) -> Dict:
        """
        Chunk a complete document with metadata preservation.

        Args:
            document: Document dictionary with text and metadata

        Returns:
            Document dictionary with chunks
        """
        text = document.get("text", "")
        if not text:
            logger.warning(f"No text found in document: {document.get('filename', 'unknown')}")
            return {**document, "chunks": [], "chunk_count": 0}

        chunks = self.chunk_text(text)
        
        # Add document metadata to each chunk
        for chunk in chunks:
            chunk.update({
                "document_id": document.get("file_id", ""),
                "filename": document.get("filename", ""),
                "source_url": document.get("file_url", ""),
                "document_metadata": document.get("metadata", {}),
            })

        result = {
            **document,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "chunking_method": chunks[0]["chunking_method"] if chunks else "none",
        }

        logger.info(
            f"Document '{document.get('filename', 'unknown')}' chunked into {len(chunks)} pieces "
            f"using {result.get('chunking_method', 'unknown')} method"
        )
        return result

    def chunk_multiple_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document dictionaries

        Returns:
            List of chunked document dictionaries
        """
        chunked_documents = []
        total_chunks = 0
        methods_used = set()

        for doc in documents:
            try:
                chunked_doc = self.chunk_document(doc)
                chunked_documents.append(chunked_doc)
                total_chunks += chunked_doc.get("chunk_count", 0)
                methods_used.add(chunked_doc.get("chunking_method", "unknown"))
            except Exception as e:
                logger.error(f"Failed to chunk document {doc.get('filename', 'unknown')}: {e}")
                # Add document with empty chunks
                chunked_documents.append({
                    **doc, 
                    "chunks": [], 
                    "chunk_count": 0, 
                    "chunking_method": "failed"
                })

        logger.info(
            f"Chunked {len(documents)} documents into {total_chunks} total chunks "
            f"using methods: {', '.join(methods_used)}"
        )
        return chunked_documents
