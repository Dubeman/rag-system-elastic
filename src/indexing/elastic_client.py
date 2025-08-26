"""Elasticsearch client setup and connection."""

import logging
from typing import Dict, Optional

from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """Elasticsearch client wrapper."""

    def __init__(self, elasticsearch_url: str = "http://localhost:9200"):
        self.elasticsearch_url = elasticsearch_url
        self.client = None
        self.connect()

    def connect(self) -> None:
        """Establish connection to Elasticsearch."""
        try:
            self.client = Elasticsearch([self.elasticsearch_url])
            
            # Test connection
            if self.client.ping():
                logger.info(f"Successfully connected to Elasticsearch at {self.elasticsearch_url}")
            else:
                raise ConnectionError("Failed to ping Elasticsearch")
                
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise

    def health_check(self) -> Dict:
        """Check Elasticsearch cluster health."""
        try:
            health = self.client.cluster.health()
            logger.info(f"Elasticsearch health: {health['status']}")
            return health
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}

    def create_index(self, index_name: str, mapping: Dict) -> bool:
        """Create index with mapping."""
        try:
            if self.client.indices.exists(index=index_name):
                logger.info(f"Index {index_name} already exists")
                return True

            response = self.client.indices.create(index=index_name, body=mapping)
            logger.info(f"Created index {index_name}: {response}")
            return True
        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            return False

    def delete_index(self, index_name: str) -> bool:
        """Delete index."""
        try:
            if self.client.indices.exists(index=index_name):
                response = self.client.indices.delete(index=index_name)
                logger.info(f"Deleted index {index_name}: {response}")
                return True
            else:
                logger.info(f"Index {index_name} does not exist")
                return True
        except Exception as e:
            logger.error(f"Failed to delete index {index_name}: {e}")
            return False
