"""FastAPI main application (v1 Elasticsearch RAG + v2 vision-native pipeline)."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import Histogram

from ..generation import LLMClient, AnswerGenerator
from ..indexing import DocumentIndexer, ElasticsearchClient
from ..ingestion import IngestionPipeline
from ..observability.metrics import (
    LLM_ERROR,
    LLM_SKIPPED,
    LLM_STUB,
    LLM_SUCCESS,
    STAGE_GENERATE,
    STAGE_INGEST,
    STAGE_RETRIEVE,
    observe_stage_seconds,
    record_ingest_chunks,
    record_llm_outcome,
    setup_prometheus_instrumentation,
)
from ..retrieval import CachedRetriever, HybridRetriever
from ..v2.pipeline_v2 import VisionPipelineV2
from .observability import log_request_summary, new_request_id, stage_timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Distinct from prometheus-fastapi-instrumentator's http_request_duration_seconds.
PIPELINE_ROUTE_LATENCY = Histogram(
    "rag_pipeline_http_duration_seconds",
    "POST /ingest and /query wall time by pipeline_version",
    ["method", "path", "pipeline_version"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

app = FastAPI(
    title="RAG System API",
    description="v1: Elasticsearch hybrid RAG. v2: vision-native page retrieval + VLM.",
    version="0.2.0",
)

es_client = None
indexer = None
ingestion_pipeline = None
retriever = None
llm_client = None
answer_generator = None
vision_pipeline_v2: Optional[VisionPipelineV2] = None


@app.on_event("startup")
async def startup_event():
    global es_client, indexer, ingestion_pipeline, retriever, llm_client, answer_generator
    global vision_pipeline_v2

    es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

    try:
        es_client = ElasticsearchClient(es_url)
        indexer = DocumentIndexer(es_client)
        ingestion_pipeline = IngestionPipeline()

        base_retriever = HybridRetriever(es_client)
        retriever = CachedRetriever(base_retriever)

        llm_client = LLMClient()
        answer_generator = AnswerGenerator(llm_client)

        v2_dir = Path(os.getenv("V2_DATA_DIR", "data/v2"))
        vision_pipeline_v2 = VisionPipelineV2(v2_dir)

        logger.info("RAG v1 + Vision v2 services initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize services: %s", e)
        raise


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = (
        request.headers.get("X-Request-ID")
        or request.headers.get("x-request-id")
        or new_request_id()
    )
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


class IngestRequest(BaseModel):
    source: str = "sample"
    folder_id: str = ""
    sample_text: str = ""
    file_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Local PDF inputs — explicit files, directories, or glob patterns; "
            "used when source='local_files'"
        ),
    )
    pipeline_version: Optional[str] = Field(
        default=None,
        description="v1 (default) or v2 — overrides PIPELINE_VERSION env",
    )


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = 5
    search_mode: str = "dense_bm25"
    generate_answer: bool = True
    pipeline_version: Optional[str] = Field(
        default=None,
        description="v1 (default) or v2 — overrides PIPELINE_VERSION env",
    )


def _resolve_pipeline_version(explicit: Optional[str]) -> str:
    v = (explicit or os.getenv("PIPELINE_VERSION", "v1") or "v1").strip().lower()
    if v not in ("v1", "v2"):
        raise HTTPException(status_code=422, detail="pipeline_version must be v1 or v2")
    return v


def _v2_llm_metric_status(llm: Optional[Dict[str, Any]]) -> str:
    if not llm:
        return LLM_SKIPPED
    st = str(llm.get("status", "")).lower()
    if st == "error":
        return LLM_ERROR
    if st == "stub":
        return LLM_STUB
    return LLM_SUCCESS


@app.get("/healthz")
async def health_check():
    try:
        es_health = es_client.health_check()
        llm_available = llm_client.is_available() if llm_client else False
        v2_ready = vision_pipeline_v2 is not None

        return {
            "status": "healthy",
            "elasticsearch": es_health["status"],
            "llm_service": "available" if llm_available else "unavailable",
            "vision_v2": "ready" if v2_ready else "unavailable",
            "services": ["elasticsearch", "api", "llm", "vision_v2"],
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")


@app.post("/ingest")
async def ingest_documents(request: IngestRequest, req: Request):
    request_id = getattr(req.state, "request_id", new_request_id())
    ingest_t0 = time.perf_counter()
    path = "/ingest"
    pv = _resolve_pipeline_version(request.pipeline_version)

    try:
        if pv == "v2":
            if vision_pipeline_v2 is None:
                raise HTTPException(status_code=503, detail="Vision v2 pipeline unavailable")

            with stage_timer(
                logger,
                request_id=request_id,
                pipeline_version=pv,
                stage="ingest_v2_total",
            ):
                if request.source == "sample":
                    sample_text = request.sample_text or (
                        "Sample document for v2 vision pipeline testing."
                    )
                    result = vision_pipeline_v2.ingest_sample_text(sample_text)
                elif request.source == "google_drive":
                    if not request.folder_id:
                        raise HTTPException(
                            status_code=400,
                            detail="folder_id required for Google Drive source",
                        )
                    result = vision_pipeline_v2.ingest_from_google_drive(
                        request.folder_id,
                        ingestion_pipeline,
                    )
                elif request.source == "local_files":
                    if not request.file_paths:
                        raise HTTPException(
                            status_code=400,
                            detail="file_paths required for local_files source",
                        )
                    result = vision_pipeline_v2.ingest_from_local_files(
                        request.file_paths,
                        ingestion_pipeline,
                    )
                else:
                    raise HTTPException(status_code=400, detail="Unsupported source type")

            elapsed = time.perf_counter() - ingest_t0
            observe_stage_seconds(STAGE_INGEST, elapsed)
            log_request_summary(
                logger,
                request_id=request_id,
                pipeline_version=pv,
                path=path,
                total_ms=elapsed * 1000.0,
                status="success",
                extra={"source": request.source},
            )
            PIPELINE_ROUTE_LATENCY.labels("POST", path, pv).observe(elapsed)

            return {
                "status": "success",
                "pipeline_version": pv,
                "source": request.source,
                "result": result,
            }

        with stage_timer(
            logger,
            request_id=request_id,
            pipeline_version=pv,
            stage="ingest_v1_total",
        ):
            if request.source == "sample":
                sample_text = request.sample_text or (
                    "This is a sample document for testing the RAG system. It contains "
                    "some text that will be chunked and indexed in Elasticsearch."
                )
                documents = ingestion_pipeline.ingest_sample_text(sample_text, "sample.txt")
            elif request.source == "google_drive":
                if not request.folder_id:
                    raise HTTPException(
                        status_code=400,
                        detail="folder_id required for Google Drive source",
                    )
                documents = ingestion_pipeline.ingest_from_google_drive(request.folder_id)
            elif request.source == "local_files":
                if not request.file_paths:
                    raise HTTPException(
                        status_code=400,
                        detail="file_paths required for local_files source",
                    )
                documents = ingestion_pipeline.ingest_from_local_files(request.file_paths)
            else:
                raise HTTPException(status_code=400, detail="Unsupported source type")

            total_indexed = 0
            total_errors = 0
            for doc in documents:
                chunks = doc.get("chunks", [])
                if chunks:
                    result = indexer.index_chunks(chunks)
                    total_indexed += result["indexed"]
                    total_errors += result["errors"]

        record_ingest_chunks(total_indexed, total_errors)
        elapsed = time.perf_counter() - ingest_t0
        observe_stage_seconds(STAGE_INGEST, elapsed)
        log_request_summary(
            logger,
            request_id=request_id,
            pipeline_version=pv,
            path=path,
            total_ms=elapsed * 1000.0,
            status="success",
            extra={"chunks_indexed": total_indexed},
        )
        PIPELINE_ROUTE_LATENCY.labels("POST", path, pv).observe(elapsed)

        return {
            "status": "success",
            "pipeline_version": pv,
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
        PIPELINE_ROUTE_LATENCY.labels("POST", path, pv).observe(time.perf_counter() - ingest_t0)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/query")
async def query_documents(request: QueryRequest, req: Request):
    request_id = getattr(req.state, "request_id", new_request_id())
    t_start = time.perf_counter()
    path = "/query"
    pv = _resolve_pipeline_version(request.pipeline_version)

    try:
        if pv == "v2":
            if vision_pipeline_v2 is None:
                raise HTTPException(status_code=503, detail="Vision v2 pipeline unavailable")

            with stage_timer(
                logger,
                request_id=request_id,
                pipeline_version=pv,
                stage="query_v2",
            ):
                response = vision_pipeline_v2.query(
                    request.question,
                    top_k=request.top_k,
                    generate_answer=request.generate_answer,
                )
            timings = response.get("timings_ms") or {}
            tr = float(timings.get("retrieve", 0)) / 1000.0
            tg = float(timings.get("generate", 0)) / 1000.0
            if tr > 0:
                observe_stage_seconds(STAGE_RETRIEVE, tr)
            if request.generate_answer and tg > 0:
                observe_stage_seconds(STAGE_GENERATE, tg)

            if request.generate_answer:
                record_llm_outcome(_v2_llm_metric_status(response.get("llm_response")))
            else:
                record_llm_outcome(LLM_SKIPPED)

            response["pipeline_version"] = pv
            response["request_id"] = request_id

            elapsed = time.perf_counter() - t_start
            log_request_summary(
                logger,
                request_id=request_id,
                pipeline_version=pv,
                path=path,
                total_ms=elapsed * 1000.0,
                status="success",
            )
            PIPELINE_ROUTE_LATENCY.labels("POST", path, pv).observe(elapsed)
            return response

        with stage_timer(
            logger,
            request_id=request_id,
            pipeline_version=pv,
            stage="retrieve_v1",
        ):
            t_retrieve_start = time.perf_counter()
            results = retriever.retrieve(
                query=request.question,
                top_k=request.top_k,
                mode=request.search_mode,
            )
        observe_stage_seconds(
            STAGE_RETRIEVE,
            time.perf_counter() - t_retrieve_start,
            request.search_mode,
        )

        response: Dict[str, Any] = {
            "question": request.question,
            "search_mode": request.search_mode,
            "pipeline_version": pv,
            "request_id": request_id,
            "results": results,
            "total_results": len(results),
            "status": "success",
        }

        if request.generate_answer and answer_generator:
            with stage_timer(
                logger,
                request_id=request_id,
                pipeline_version=pv,
                stage="generate_v1",
            ):
                t_gen_start = time.perf_counter()
                try:
                    llm_response = answer_generator.generate_with_citations(
                        request.question, results
                    )
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
                observe_stage_seconds(
                    STAGE_GENERATE,
                    time.perf_counter() - t_gen_start,
                    request.search_mode,
                )
        else:
            record_llm_outcome(LLM_SKIPPED)

        elapsed = time.perf_counter() - t_start
        log_request_summary(
            logger,
            request_id=request_id,
            pipeline_version=pv,
            path=path,
            total_ms=elapsed * 1000.0,
            status="success",
        )
        PIPELINE_ROUTE_LATENCY.labels("POST", path, pv).observe(elapsed)
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Enhanced query failed: %s", e)
        PIPELINE_ROUTE_LATENCY.labels("POST", path, pv).observe(time.perf_counter() - t_start)
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


setup_prometheus_instrumentation(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
