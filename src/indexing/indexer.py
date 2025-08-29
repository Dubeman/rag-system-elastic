"""Document indexing operations."""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .elastic_client import ElasticsearchClient

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Handles document indexing to Elasticsearch."""

    def __init__(self, es_client: ElasticsearchClient, index_name: str = "rag_documents"):
        self.es_client = es_client
        self.index_name = index_name
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model successfully")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self.embedding_model = None
            
        self.ensure_index_exists()

    def get_index_mapping(self) -> Dict:
        """Get the index mapping configuration for hybrid search."""
        return {
            "mappings": {
                "properties": {
                    # Text fields for BM25 search
                    "text": {"type": "text"},
                    "content": {"type": "text"},  # Alias for compatibility
                    
                    # Dense embeddings for semantic search
                    "dense_embedding": {
                        "type": "dense_vector",
                        "dims": 384,  # all-MiniLM-L6-v2 dimensions
                        "index": True,
                        "similarity": "cosine"
                    },
                    
                    # ELSER sparse embeddings (when available)
                    "text_expansion": {
                        "type": "sparse_vector",
                        "dims": 30522  # ELSER vocabulary size
                    },
                    
                    # Metadata fields
                    "chunk_id": {"type": "integer"},
                    "document_id": {"type": "keyword"},
                    "filename": {"type": "keyword"},
                    "source_url": {"type": "keyword"},
                    "file_url": {"type": "keyword"},  # Added for compatibility
                    "modified_time": {"type": "keyword"},
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

    def generate_dense_embedding(self, text: str) -> Optional[List[float]]:
        """Generate dense embeddings using sentence-transformers."""
        if not self.embedding_model:
            return None
            
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def generate_elser_embedding(self, text: str) -> Optional[Dict]:
        """Generate ELSER sparse embeddings using Elasticsearch ML model."""
        try:
            # ELSER expects the text in a specific field format
            response = self.es_client.client.ml.infer_trained_model(
                model_id=".elser_model_2",
                docs=[{"text_field": text}],  # Use "text_field" as expected by ELSER
                timeout="15s"  # Add timeout to prevent hanging
            )
            
            # ELSER response structure:
            # {
            #   "inference_results": [
            #     {
            #       "predicted_value": {
            #         "text_expansion": {
            #           "tokens": {"token1": 1.23, "token2": 0.45, ...}
            #         }
            #       }
            #     }
            #   ]
            # }
            
            if response and "inference_results" in response:
                # Get the first (and only) result
                result = response["inference_results"][0]
                
                # Extract the text_expansion tokens
                if "predicted_value" in result and "text_expansion" in result["predicted_value"]:
                    return result["predicted_value"]["text_expansion"]
            
            logger.warning("ELSER response structure unexpected")
            return None
            
        except Exception as e:
            logger.warning(f"ELSER embedding failed for chunk (continuing without it): {e}")
            return None  # Continue indexing without ELSER

    def generate_elser_embeddings_batch(self, texts: List[str]) -> List[Optional[Dict]]:
        """Generate ELSER embeddings for multiple texts in one call."""
        try:
            # Prepare batch of documents
            docs = [{"text_field": text} for text in texts]
            
            # Single API call for all texts
            response = self.es_client.client.ml.infer_trained_model(
                model_id=".elser_model_2",
                docs=docs,
                timeout="60s"  # Longer timeout for batch
            )
            
            results = []
            if response and "inference_results" in response:
                for result in response["inference_results"]:
                    if "predicted_value" in result and "text_expansion" in result["predicted_value"]:
                        results.append(result["predicted_value"]["text_expansion"])
                    else:
                        results.append(None)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate ELSER embeddings batch: {e}")
            return [None] * len(texts)

    def index_chunks(self, chunks: List[Dict]) -> Dict:
        """Index document chunks with batch ELSER processing."""
        if not chunks:
            return {"indexed": 0, "errors": 0}

        # Extract all texts for batch processing
        texts = [chunk.get("text", "") for chunk in chunks]
        
        # Generate all ELSER embeddings in one call
        logger.info(f"Generating ELSER embeddings for {len(chunks)} chunks in batch...")
        elser_embeddings = self.generate_elser_embeddings_batch(texts)
        logger.info(f"ELSER batch processing complete: {len([e for e in elser_embeddings if e])} successful, {len([e for e in elser_embeddings if not e])} failed")

        indexed_count = 0
        error_count = 0

        for i, chunk in enumerate(chunks):
            try:
                doc_id = f"{chunk.get('document_id', 'unknown')}_{chunk.get('chunk_id', 0)}"
                text_content = chunk.get("text", "")
                
                # Generate dense embedding
                dense_embedding = self.generate_dense_embedding(text_content)
                
                # Build document with all fields
                doc = {
                    # Text content for BM25
                    "text": text_content,
                    "content": text_content,  # Alias for compatibility
                    
                    # Embeddings
                    "dense_embedding": dense_embedding,
                    
                    # Metadata
                    "chunk_id": chunk.get("chunk_id", 0),
                    "document_id": chunk.get("document_id", ""),
                    "filename": chunk.get("filename", ""),
                    "source_url": chunk.get("source_url", ""),
                    "file_url": chunk.get("file_url", chunk.get("source_url", "")),
                    "modified_time": chunk.get("modified_time", ""),
                    "token_count": chunk.get("token_count", 0),
                    "char_count": chunk.get("char_count", 0),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                
                # Add ELSER embedding if available
                if elser_embeddings[i]:
                    doc["text_expansion"] = elser_embeddings[i]
                    logger.debug(f"Added ELSER embedding with {len(elser_embeddings[i].get('predicted_value', {}))} tokens")

                response = self.es_client.client.index(
                    index=self.index_name,
                    id=doc_id,
                    document=doc
                )
                
                if response["result"] in ["created", "updated"]:
                    indexed_count += 1
                    if indexed_count % 10 == 0:  # Log progress
                        logger.info(f"Indexed {indexed_count} chunks so far...")
                else:
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to index chunk {chunk.get('chunk_id', 'unknown')}: {e}")
                error_count += 1

        logger.info(f"Enhanced indexing complete: {indexed_count} chunks indexed, {error_count} errors")
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

    def test_elser_connection(self) -> bool:
        """Test if ELSER model is accessible and working."""
        try:
            test_text = "This is a test document for ELSER."
            embedding = self.generate_elser_embedding(test_text)
            
            if embedding and "tokens" in embedding:
                token_count = len(embedding["tokens"])
                logger.info(f"ELSER test successful: generated {token_count} tokens")
                return True
            else:
                logger.warning("ELSER test failed: no valid embedding generated")
                return False
                
        except Exception as e:
            logger.error(f"ELSER test failed with error: {e}")
            return False
