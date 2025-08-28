"""Cached retriever for immediate performance boost."""

import hashlib
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class CachedRetriever:
    """Simple in-memory cache for immediate performance boost."""
    
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever
        self.cache = {}  # Simple dict cache
        self.cache_ttl = 300  # 5 minutes
    
    def get_cache_key(self, query: str, mode: str, top_k: int) -> str:
        """Generate cache key for query."""
        return f"{query}_{mode}_{top_k}"
    
    def get_cached_results(self, query: str, mode: str, top_k: int) -> Optional[List[Dict]]:
        """Get cached results if available."""
        cache_key = self.get_cache_key(query, mode, top_k)
        if cache_key in self.cache:
            logger.info(f"Cache HIT for query: {query[:50]}...")
            return self.cache[cache_key]
        return None
    
    def cache_results(self, query: str, mode: str, top_k: int, results: List[Dict]):
        """Cache results for future use."""
        cache_key = self.get_cache_key(query, mode, top_k)
        self.cache[cache_key] = results
        logger.info(f"Cached results for query: {query[:50]}...")
    
    def retrieve(self, query: str, top_k: int = 5, mode: str = "dense_bm25") -> List[Dict]:
        """Main retrieval method with caching."""
        
        # Try cache first
        cached = self.get_cached_results(query, mode, top_k)
        if cached:
            return cached
        
        # Cache miss - retrieve from base retriever
        logger.info(f"Cache MISS for query: {query[:50]}...")
        results = self.base_retriever.retrieve(query, top_k, mode)
        
        # Cache the results
        self.cache_results(query, mode, top_k, results)
        
        return results
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "cache_ttl": self.cache_ttl
        }
