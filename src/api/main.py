"""FastAPI main application."""

import logging
import os
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


from ..indexing import DocumentIndexer, ElasticsearchClient
from ..ingestion import IngestionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation API with Elasticsearch",
    version="0.1.0"
)

# Global clients
es_client = None
indexer = None
ingestion_pipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global es_client, indexer, ingestion_pipeline
    
    es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    
    try:
        es_client = ElasticsearchClient(es_url)
        indexer = DocumentIndexer(es_client)
        ingestion_pipeline = IngestionPipeline()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


class IngestRequest(BaseModel):
    source: str = "sample"
    folder_id: str = ""
    sample_text: str = ""


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    try:
        es_health = es_client.health_check()
        return {
            "status": "healthy",
            "elasticsearch": es_health["status"],
            "services": ["elasticsearch", "api"]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")


@app.post("/ingest")
async def ingest_documents(request: IngestRequest):
    """Ingest documents endpoint."""
    try:
        if request.source == "sample":
            # For testing with sample text
            sample_text = request.sample_text or "This is a sample document for testing the RAG system. It contains some text that will be chunked and indexed in Elasticsearch."
            
            documents = ingestion_pipeline.ingest_sample_text(sample_text, "sample.txt")
            
        elif request.source == "google_drive":
            if not request.folder_id:
                raise HTTPException(status_code=400, detail="folder_id required for Google Drive source")
            
            documents = ingestion_pipeline.ingest_from_google_drive(request.folder_id)
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported source type")

        # Index all chunks
        total_indexed = 0
        total_errors = 0
        
        for doc in documents:
            chunks = doc.get("chunks", [])
            if chunks:
                result = indexer.index_chunks(chunks)
                total_indexed += result["indexed"]
                total_errors += result["errors"]

        return {
            "status": "success",
            "documents_processed": len(documents),
            "chunks_indexed": total_indexed,
            "errors": total_errors,
            "source": request.source
        }
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query documents endpoint (basic implementation)."""
    try:
        results = indexer.search_basic(request.question, request.top_k)
        
        return {
            "question": request.question,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)