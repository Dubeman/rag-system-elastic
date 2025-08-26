"""Document indexing operations."""

import logging
from datetime import datetime
from typing import Dict, List

from .elastic_client import ElasticsearchClient

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Handles document indexing to Elasticsearch."""

    def __init__(self, es_client: ElasticsearchClient, index_name: str = "rag_documents"):
        self.es_client = es_client
        self.index_name = index_name
        self.ensure_index_exists()

    def get_index_mapping(self) -> Dict:
        """Get the index mapping configuration."""
        return {
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "chunk_id": {"type": "integer"},
                    "document_id": {"type": "keyword"},
                    "filename": {"type": "keyword"},
                    "source_url": {"type": "keyword"},
                    "token_count": {"type": "integer"},
                    "char_count": {"type": "integer"},
                    "timestamp": {"type": "date"},
                }
            }
        }

    def ensure_index_exists(self) -> None:
        """Ensure the index exists with proper mapping."""
        mapping = self.get_index_mapping()
        self.es_client.create_index(self.index_name, mapping)

    def index_chunks(self, chunks: List[Dict]) -> Dict:
        """Index document chunks."""
        if not chunks:
            return {"indexed": 0, "errors": 0}

        indexed_count = 0
        error_count = 0

        for chunk in chunks:
            try:
                doc_id = f"{chunk.get('document_id', 'unknown')}_{chunk.get('chunk_id', 0)}"
                
                doc = {
                    "text": chunk.get("text", ""),
                    "chunk_id": chunk.get("chunk_id", 0),
                    "document_id": chunk.get("document_id", ""),
                    "filename": chunk.get("filename", ""),
                    "source_url": chunk.get("source_url", ""),
                    "token_count": chunk.get("token_count", 0),
                    "char_count": chunk.get("char_count", 0),
                    "timestamp": datetime.utcnow().isoformat(),
                }

                response = self.es_client.client.index(
                    index=self.index_name,
                    id=doc_id,
                    document=doc
                )
                
                if response["result"] in ["created", "updated"]:
                    indexed_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to index chunk {chunk.get('chunk_id', 'unknown')}: {e}")
                error_count += 1

        logger.info(f"Indexed {indexed_count} chunks, {error_count} errors")
        return {"indexed": indexed_count, "errors": error_count}

    def search_basic(self, query: str, size: int = 5) -> List[Dict]:
        """Basic text search."""
        try:
            response = self.es_client.client.search(
                index=self.index_name,
                body={
                    "query": {"match": {"text": query}},
                    "size": size
                }
            )
            
            hits = response["hits"]["hits"]
            results = []
            for hit in hits:
                results.append({
                    "text": hit["_source"]["text"],
                    "filename": hit["_source"]["filename"],
                    "score": hit["_score"],
                })
            
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
