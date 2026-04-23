"""Ingestion pipeline orchestrating all steps with public Google Drive support."""

import glob as globlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .enhanced_chunker import EnhancedChunker
from .pdf_parser import PDFParser

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates document ingestion from source to chunks."""

    def __init__(
        self,
        google_credentials_path: Optional[str] = None,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ):
        self.google_credentials_path = google_credentials_path
        self.pdf_parser = PDFParser()
        self.chunker = EnhancedChunker(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            use_langchain=True
        )
        self.google_client = None

    def initialize_google_client(self) -> None:
        """Initialize Google Drive client with fallback for public folders."""
        try:
            if self.google_credentials_path and Path(self.google_credentials_path).exists():
                from .fetch_drive import GoogleDriveClient
                self.google_client = GoogleDriveClient(self.google_credentials_path)
                logger.info("Google Drive client initialized with credentials")
            else:
                # Fallback to public client for public folders
                from .public_drive_client import PublicGoogleDriveClient
                self.google_client = PublicGoogleDriveClient()
                logger.info("Google Drive client initialized for public access")
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive client: {e}")
            # Try public client as last resort
            try:
                from .public_drive_client import PublicGoogleDriveClient
                self.google_client = PublicGoogleDriveClient()
                logger.info("Fallback to public Google Drive client")
            except Exception as e2:
                logger.error(f"All Google Drive client initialization failed: {e2}")
                self.google_client = None

    def ingest_from_google_drive(self, folder_id: str) -> List[Dict]:
        """Ingest documents from Google Drive folder."""
        if not self.google_client:
            self.initialize_google_client()

        if not self.google_client:
            # Try one more time with public client
            try:
                from .public_drive_client import PublicGoogleDriveClient
                self.google_client = PublicGoogleDriveClient()
                logger.info("Using public Google Drive client for folder access")
            except Exception as e:
                raise ValueError(f"Google Drive access failed: {e}")

        # Fetch PDFs (works with both authenticated and public clients)
        try:
            if hasattr(self.google_client, 'fetch_pdfs_from_public_folder'):
                # Public client
                pdf_files = self.google_client.fetch_pdfs_from_public_folder(folder_id)
            else:
                # Authenticated client  
                pdf_files = self.google_client.fetch_pdfs_from_folder(folder_id)
                
            logger.info(f"Fetched {len(pdf_files)} PDF files from Google Drive")
        except Exception as e:
            logger.error(f"Failed to fetch PDFs: {e}")
            raise

        # Parse PDFs
        parsed_docs = self.pdf_parser.parse_multiple_pdfs(pdf_files)
        successful_docs = [doc for doc in parsed_docs if doc["extraction_success"]]
        logger.info(f"Successfully parsed {len(successful_docs)} documents")

        # Chunk documents
        chunked_docs = []
        for doc in successful_docs:
            chunked_doc = self.chunker.chunk_document(doc)
            chunked_docs.append(chunked_doc)

        total_chunks = sum(doc.get("chunk_count", 0) for doc in chunked_docs)
        logger.info(f"Created {total_chunks} chunks from {len(chunked_docs)} documents")

        return chunked_docs

    def ingest_from_local_files(self, file_paths: List[str]) -> List[Dict]:
        """Ingest PDFs from local filesystem paths.

        Args:
            file_paths: List of absolute or relative paths to PDF files.

        Returns:
            List of chunked document dicts (same shape as ingest_from_google_drive).
        """
        resolved_paths = self._expand_local_pdf_paths(file_paths)
        pdf_files = []
        for path in resolved_paths:
            try:
                content = path.read_bytes()
                pdf_files.append({
                    "id": path.stem,
                    "name": path.name,
                    "content": content,
                    "size": len(content),
                    "url": str(path.resolve()),
                    "modified_time": "",
                    "mime_type": "application/pdf",
                })
                logger.info("Loaded local PDF: %s (%d bytes)", path.name, len(content))
            except Exception as e:
                logger.error("Failed to read %s: %s", str(path), e)

        if not pdf_files:
            logger.warning("No valid PDFs found in provided file_paths")
            return []

        parsed_docs = self.pdf_parser.parse_multiple_pdfs(pdf_files)
        successful_docs = [doc for doc in parsed_docs if doc["extraction_success"]]
        logger.info("Parsed %d / %d documents successfully", len(successful_docs), len(pdf_files))

        chunked_docs = []
        for doc in successful_docs:
            chunked_doc = self.chunker.chunk_document(doc)
            chunked_docs.append(chunked_doc)

        total_chunks = sum(doc.get("chunk_count", 0) for doc in chunked_docs)
        logger.info("Created %d chunks from %d documents", total_chunks, len(chunked_docs))
        return chunked_docs

    def _expand_local_pdf_paths(self, file_paths: List[str]) -> List[Path]:
        """Expand explicit paths, directories, and globs to deduplicated PDF paths."""
        expanded_paths: List[Path] = []
        seen = set()
        raw_count = len(file_paths)

        for raw_path in file_paths:
            if not raw_path:
                logger.warning("Empty path entry in file_paths, skipping")
                continue

            path = Path(raw_path)
            has_glob = any(ch in raw_path for ch in "*?[]")

            candidates: List[Path] = []
            if has_glob:
                candidates = [Path(p) for p in globlib.glob(raw_path)]
                if not candidates:
                    logger.warning("Glob matched no files: %s", raw_path)
            elif path.is_dir():
                candidates = sorted(path.glob("*.pdf"))
                if not candidates:
                    logger.warning("Directory has no PDF files: %s", raw_path)
            else:
                candidates = [path]

            for candidate in candidates:
                if not candidate.exists():
                    logger.warning("File not found, skipping: %s", str(candidate))
                    continue
                if candidate.is_dir():
                    logger.warning("Directory path not allowed here, skipping: %s", str(candidate))
                    continue
                if candidate.suffix.lower() != ".pdf":
                    logger.warning("Non-PDF file skipped: %s", str(candidate))
                    continue

                dedupe_key = str(candidate.resolve())
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                expanded_paths.append(candidate)

        logger.info(
            "Expanded %d path input(s) to %d unique PDF file(s)",
            raw_count,
            len(expanded_paths),
        )
        return expanded_paths

    def ingest_sample_text(self, text: str, filename: str = "sample.txt") -> List[Dict]:
        """Ingest sample text for testing."""
        sample_doc = {
            "file_id": "sample_id",
            "filename": filename,
            "text": text,
            "file_url": "",
            "extraction_success": True,
        }

        chunked_doc = self.chunker.chunk_document(sample_doc)
        logger.info(f"Created sample document with {chunked_doc.get('chunk_count', 0)} chunks")
        return [chunked_doc]
