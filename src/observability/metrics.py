"""Prometheus metrics for the RAG API (bounded labels only)."""

import os
from typing import Optional

from fastapi import FastAPI
from prometheus_client import Counter, Histogram

# Stages: retrieve, generate, ingest (wall time for POST /ingest)
STAGE_RETRIEVE = "retrieve"
STAGE_GENERATE = "generate"
STAGE_INGEST = "ingest"

VALID_SEARCH_MODES = frozenset(
    {
        "bm25_only",
        "dense_only",
        "elser_only",
        "dense_bm25",
        "full_hybrid",
    }
)

LLM_SUCCESS = "success"
LLM_ERROR = "error"
LLM_SKIPPED = "skipped"
LLM_STUB = "stub"

RAG_STAGE_DURATION = Histogram(
    "rag_stage_duration_seconds",
    "Time spent in RAG pipeline stages",
    ["stage", "search_mode"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

RAG_LLM_GENERATION = Counter(
    "rag_llm_generation_total",
    "LLM answer generation outcomes",
    ["status"],
)

RAG_CACHE = Counter(
    "rag_cache_total",
    "Retrieval cache hits and misses",
    ["result"],
)

RAG_INGEST_CHUNKS = Counter(
    "rag_ingest_chunks_total",
    "Chunks successfully indexed during ingest",
)

RAG_INGEST_ERRORS = Counter(
    "rag_ingest_errors_total",
    "Chunk indexing errors during ingest",
)


def metrics_enabled() -> bool:
    return os.getenv("METRICS_ENABLED", "true").lower() in ("1", "true", "yes")


def normalize_search_mode(mode: Optional[str]) -> str:
    if not mode or mode not in VALID_SEARCH_MODES:
        return "unknown"
    return mode


def observe_stage_seconds(stage: str, seconds: float, search_mode: Optional[str] = None) -> None:
    if not metrics_enabled():
        return
    if stage == STAGE_INGEST:
        sm = "none"
    else:
        sm = normalize_search_mode(search_mode)
    RAG_STAGE_DURATION.labels(stage=stage, search_mode=sm).observe(seconds)


def record_llm_outcome(status: str) -> None:
    if not metrics_enabled():
        return
    if status not in (LLM_SUCCESS, LLM_ERROR, LLM_SKIPPED, LLM_STUB):
        status = LLM_ERROR
    RAG_LLM_GENERATION.labels(status=status).inc()


def record_cache_result(hit: bool) -> None:
    if not metrics_enabled():
        return
    RAG_CACHE.labels(result="hit" if hit else "miss").inc()


def record_ingest_chunks(indexed: int, errors: int) -> None:
    if not metrics_enabled():
        return
    if indexed:
        RAG_INGEST_CHUNKS.inc(indexed)
    if errors:
        RAG_INGEST_ERRORS.inc(errors)


def setup_prometheus_instrumentation(app: FastAPI) -> None:
    """HTTP RED metrics and /metrics endpoint (no-op if METRICS_ENABLED=false)."""
    if not metrics_enabled():
        return
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],
    ).instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")
