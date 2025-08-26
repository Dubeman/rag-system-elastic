#!/usr/bin/env python3
"""Patch to add public Google Drive client fallback."""

# Updated initialize_google_client method
def initialize_google_client_updated(self) -> None:
    """Initialize Google Drive client with fallback for public folders."""
    try:
        if self.google_credentials_path:
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

# Updated ingest_from_google_drive method
def ingest_from_google_drive_updated(self, folder_id: str) -> List[Dict]:
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

print("Pipeline patch created - this shows the modifications needed")
