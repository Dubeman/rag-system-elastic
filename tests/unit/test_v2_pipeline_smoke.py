"""Smoke tests for v2 vision pipeline without Elasticsearch or FastAPI."""

import os
from pathlib import Path

import pytest

# Ensure mock embeddings for CI
os.environ.setdefault("COLPALI_USE_MOCK", "true")
os.environ.setdefault("VECTOR_BACKEND", "faiss")


def test_v2_ingest_sample_and_query(tmp_path: Path):
    from v2.pipeline_v2 import VisionPipelineV2

    data = tmp_path / "v2data"
    os.environ["V2_DATA_DIR"] = str(data)
    pipe = VisionPipelineV2(data)
    ing = pipe.ingest_sample_text(
        "Hello world. This is a smoke test document for vision v2."
    )
    assert ing.get("pages_rendered", 0) >= 1
    assert ing.get("vectors_indexed", 0) >= 1

    out = pipe.query("What is this document about?", top_k=3, generate_answer=True)
    assert out["status"] == "success"
    assert out["pipeline_version"] == "v2"
    assert "timings_ms" in out
    assert out["total_results"] >= 1
    llm = out.get("llm_response") or {}
    assert llm.get("status") in ("stub", "ok", "error")
