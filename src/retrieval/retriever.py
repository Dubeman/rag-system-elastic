"""Hybrid retrieval implementation."""

import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

from ..indexing.elastic_client import ElasticsearchClient

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining BM25, dense embeddings, and ELSER."""

    def __init__(self, es_client: ElasticsearchClient, index_name: str = "rag_documents"):
        self.es_client = es_client
        self.index_name = index_name
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model for retrieval")
        except Exception as e:
            logger.warning(f"Failed to load embedding model for retrieval: {e}")
            self.embedding_model = None

    def generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for the query."""
        if not self.embedding_model:
            return None
            
        try:
            embedding = self.embedding_model.encode(query, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return None

    def search_bm25(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 text search."""
        try:
            response = self.es_client.client.search(
                index=self.index_name,
                body={
                    "query": {
                        "bool": {
                            "should": [
                                {"match": {"text": {"query": query, "boost": 1.0}}},
                                {"match": {"content": {"query": query, "boost": 1.0}}}
                            ]
                        }
                    },
                    "size": top_k,
                    "_source": ["text", "content", "filename", "chunk_id", "file_url", "modified_time"]
                }
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "content": hit["_source"].get("text", hit["_source"].get("content", "")),
                    "filename": hit["_source"].get("filename", ""),
                    "chunk_id": hit["_source"].get("chunk_id", ""),
                    "file_url": hit["_source"].get("file_url", ""),
                    "modified_time": hit["_source"].get("modified_time", ""),
                    "_score": hit["_score"],
                    "_source": hit["_source"],
                    "search_type": "bm25"
                })
            
            logger.info(f"BM25 search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def search_dense(self, query: str, top_k: int = 10) -> List[Dict]:
        """Dense vector similarity search."""
        if not self.embedding_model:
            logger.warning("No embedding model available for dense search")
            return []
            
        query_embedding = self.generate_query_embedding(query)
        if not query_embedding:
            return []
        
        try:
            # Use the correct knn search syntax for Elasticsearch 8.x
            response = self.es_client.client.search(
                index=self.index_name,
                body={
                    "knn": {
                        "field": "dense_embedding",
                        "query_vector": query_embedding,
                        "k": top_k,
                        "num_candidates": min(top_k * 10, 1000)
                    },
                    "query": {
                        "bool": {
                            "filter": [
                                {"exists": {"field": "dense_embedding"}}
                            ]
                        }
                    },
                    "size": top_k,
                    "_source": ["text", "content", "filename", "chunk_id", "file_url", "modified_time"]
                }
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "content": hit["_source"].get("text", hit["_source"].get("content", "")),
                    "filename": hit["_source"].get("filename", ""),
                    "chunk_id": hit["_source"].get("chunk_id", ""),
                    "file_url": hit["_source"].get("file_url", ""),
                    "modified_time": hit["_source"].get("modified_time", ""),
                    "_score": hit["_score"],
                    "_source": hit["_source"],
                    "search_type": "dense"
                })
            
            logger.info(f"Dense search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []

    def search_elser(self, query: str, top_k: int = 10) -> List[Dict]:
        """ELSER sparse vector search using the inference pipeline."""
        try:
            # Generate ELSER embedding for the query
            query_embedding = self._generate_elser_query_embedding(query)
            if not query_embedding:
                logger.warning("Failed to generate ELSER query embedding")
                return []
            
            # Search using the text_expansion field
            response = self.es_client.client.search(
                index=self.index_name,
                body={
                    "query": {
                        "text_expansion": {
                            "text_expansion": {
                                "query_expansion": query_embedding,
                                "model_id": ".elser_model_2"
                            }
                        }
                    },
                    "size": top_k,
                    "_source": ["text", "content", "filename", "chunk_id", "file_url", "modified_time", "text_expansion"]
                }
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "content": hit["_source"].get("text", hit["_source"].get("content", "")),
                    "filename": hit["_source"].get("filename", ""),
                    "chunk_id": hit["_source"].get("chunk_id", ""),
                    "file_url": hit["_source"].get("file_url", ""),
                    "modified_time": hit["_source"].get("modified_time", ""),
                    "_score": hit["_score"],
                    "_source": hit["_source"],
                    "search_type": "elser"
                })
            
            logger.info(f"ELSER search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"ELSER search failed: {e}")
            return []

    def _generate_elser_query_embedding(self, query: str) -> Optional[Dict]:
        """Generate ELSER embedding for the query using the inference pipeline."""
        try:
            # Use the ELSER pipeline to generate query embedding
            response = self.es_client.client.ingest.simulate(
                id="elser_pipeline",
                body={
                    "docs": [
                        {
                            "_source": {
                                "text": query
                            }
                        }
                    ]
                }
            )
            
            if response and "docs" in response:
                first_doc = response["docs"][0]
                if "_source" in first_doc and "text_expansion" in first_doc["_source"]:
                    return first_doc["_source"]["text_expansion"]
            
            logger.warning("Failed to generate ELSER query embedding")
            return None
            
        except Exception as e:
            logger.error(f"ELSER query embedding generation failed: {e}")
            return None

    def reciprocal_rank_fusion(self, results_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
        """Combine multiple result lists using Reciprocal Rank Fusion."""
        if not results_lists:
            return []
        
        # Collect all unique documents by their document ID
        doc_scores = defaultdict(float)
        doc_info = {}
        
        for results in results_lists:
            for rank, result in enumerate(results, 1):
                # Create unique document identifier
                doc_id = f"{result.get('filename', '')}_{result.get('chunk_id', '')}"
                
                # Calculate RRF score: 1 / (k + rank)
                rrf_score = 1.0 / (k + rank)
                doc_scores[doc_id] += rrf_score
                
                # Store document info (use the first occurrence)
                if doc_id not in doc_info:
                    doc_info[doc_id] = result
        
        # Sort by combined RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final results
        fused_results = []
        for doc_id, score in sorted_docs:
            result = doc_info[doc_id].copy()
            result["_score"] = score
            result["search_type"] = "hybrid_rrf"
            fused_results.append(result)
        
        logger.info(f"RRF fusion produced {len(fused_results)} results from {len(results_lists)} search methods")
        return fused_results

    def search_hybrid(self, query: str, top_k: int = 5, mode: str = "dense_bm25") -> List[Dict]:
        """Perform hybrid search with configurable modes."""
        all_results = []
        
        if mode == "bm25_only":
            return self.search_bm25(query, top_k)
        
        elif mode == "dense_only":
            return self.search_dense(query, top_k)[:top_k]
        
        elif mode == "elser_only":
            return self.search_elser(query, top_k)
        
        elif mode == "dense_bm25":
            # Dense + BM25 hybrid
            bm25_results = self.search_bm25(query, top_k * 2)
            dense_results = self.search_dense(query, top_k * 2)
            
            if bm25_results:
                all_results.append(bm25_results)
            if dense_results:
                all_results.append(dense_results)
                
        elif mode == "full_hybrid":
            # All three: ELSER + Dense + BM25
            bm25_results = self.search_bm25(query, top_k * 2)
            dense_results = self.search_dense(query, top_k * 2)
            elser_results = self.search_elser(query, top_k * 2)
            
            if bm25_results:
                all_results.append(bm25_results)
            if dense_results:
                all_results.append(dense_results)
            if elser_results:
                all_results.append(elser_results)
        
        else:
            logger.warning(f"Unknown search mode: {mode}, falling back to dense_bm25")
            return self.search_hybrid(query, top_k, "dense_bm25")
        
        # Fuse results and return top_k
        if len(all_results) > 1:
            fused = self.reciprocal_rank_fusion(all_results)
            return fused[:top_k]
        elif len(all_results) == 1:
            return all_results[0][:top_k]
        else:
            logger.warning("No search results from any method")
            return []

    def retrieve(self, query: str, top_k: int = 5, mode: str = "dense_bm25") -> List[Dict]:
        """Main retrieval method."""
        logger.info(f"Retrieving with mode='{mode}', top_k={top_k}, query='{query[:50]}...'")
        
        try:
            results = self.search_hybrid(query, top_k, mode)
            
            # Format results for consistency
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append({
                    "content": result.get("content", ""),
                    "filename": result.get("filename", "Unknown"),
                    "chunk_id": result.get("chunk_id", "N/A"),
                    "file_url": result.get("file_url", "N/A"),
                    "modified_time": result.get("modified_time", "N/A"),
                    "_score": result.get("_score", 0.0),
                    "_source": result.get("_source", {}),
                    "rank": i + 1,
                    "search_type": result.get("search_type", mode)
                })
            
            logger.info(f"Retrieved {len(formatted_results)} results successfully")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
