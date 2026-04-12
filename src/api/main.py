"""FastAPI main application."""

import logging
import os
import time
import uuid
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware

from ..indexing import DocumentIndexer, ElasticsearchClient
from ..ingestion import IngestionPipeline
from ..retrieval import HybridRetriever, CachedRetriever
from ..generation import LLMClient, AnswerGenerator
from ..observability.metrics import (
    LLM_ERROR,
    LLM_SKIPPED,
    LLM_SUCCESS,
    STAGE_GENERATE,
    STAGE_INGEST,
    STAGE_RETRIEVE,
    observe_stage_seconds,
    record_ingest_chunks,
    record_llm_outcome,
    setup_prometheus_instrumentation,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation API with Elasticsearch",
    version="0.1.0",
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Propagate X-Request-ID for log and trace correlation."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


app.add_middleware(RequestIDMiddleware)

# Global clients
es_client = None
indexer = None
ingestion_pipeline = None
retriever = None
llm_client = None
answer_generator = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global es_client, indexer, ingestion_pipeline, retriever, llm_client, answer_generator

    es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

    try:
        es_client = ElasticsearchClient(es_url)
        indexer = DocumentIndexer(es_client)
        ingestion_pipeline = IngestionPipeline()

        # Create base retriever and wrap with cache
        base_retriever = HybridRetriever(es_client)
        retriever = CachedRetriever(base_retriever)

        # Initialize LLM components
        llm_client = LLMClient()
        answer_generator = AnswerGenerator(llm_client)

        logger.info("Enhanced RAG services with LLM initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize services: %s", e)
        raise


class IngestRequest(BaseModel):
    source: str = "sample"
    folder_id: str = ""
    sample_text: str = ""


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    search_mode: str = "dense_bm25"  # Options: bm25_only, dense_only, elser_only, dense_bm25, full_hybrid
    generate_answer: bool = True  # Whether to generate LLM answer


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    try:
        es_health = es_client.health_check()
        llm_available = llm_client.is_available() if llm_client else False

        return {
            "status": "healthy",
            "elasticsearch": es_health["status"],
            "llm_service": "available" if llm_available else "unavailable",
            "services": ["elasticsearch", "api", "llm"],
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")


@app.post("/ingest")
async def ingest_documents(request: IngestRequest):
    """Ingest documents endpoint."""
    ingest_t0 = time.perf_counter()
    try:
        if request.source == "sample":
            # For testing with sample text
            sample_text = request.sample_text or (
                "This is a sample document for testing the RAG system. It contains some text that will be "
                "chunked and indexed in Elasticsearch."
            )

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

        record_ingest_chunks(total_indexed, total_errors)
        observe_stage_seconds(STAGE_INGEST, time.perf_counter() - ingest_t0)

        return {
            "status": "success",
            "documents_processed": len(documents),
            "chunks_indexed": total_indexed,
            "errors": total_errors,
            "source": request.source,
        }

    except HTTPException:
        observe_stage_seconds(STAGE_INGEST, time.perf_counter() - ingest_t0)
        raise
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        observe_stage_seconds(STAGE_INGEST, time.perf_counter() - ingest_t0)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/query")
async def query_documents(request: QueryRequest):
    """Enhanced query endpoint with hybrid search and LLM generation."""
    try:
        t_retrieve_start = time.perf_counter()
        results = retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
            mode=request.search_mode,
        )
        observe_stage_seconds(STAGE_RETRIEVE, time.perf_counter() - t_retrieve_start, request.search_mode)

        response: Dict = {
            "question": request.question,
            "search_mode": request.search_mode,
            "results": results,
            "total_results": len(results),
            "status": "success",
        }

        if request.generate_answer and answer_generator:
            t_gen_start = time.perf_counter()
            try:
                llm_response = answer_generator.generate_with_citations(request.question, results)
                response["llm_response"] = llm_response
                record_llm_outcome(LLM_SUCCESS)
                logger.info(
                    "LLM answer generated for query: %s...",
                    request.question[:50],
                )
            except Exception as llm_error:
                logger.warning("LLM generation failed: %s", llm_error)
                record_llm_outcome(LLM_ERROR)
                response["llm_response"] = {
                    "answer": "LLM service temporarily unavailable.",
                    "citations": [],
                    "status": "error",
                    "error": str(llm_error),
                }
            observe_stage_seconds(STAGE_GENERATE, time.perf_counter() - t_gen_start, request.search_mode)
        else:
            record_llm_outcome(LLM_SKIPPED)

        return response

    except Exception as e:
        logger.error("Enhanced query failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


setup_prometheus_instrumentation(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
