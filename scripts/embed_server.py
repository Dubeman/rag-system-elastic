#!/usr/bin/env python3
"""
Minimal HTTP server for ColPali embeddings (RunPod or local GPU).

Unset COLPALI_EMBED_URL on this process; set COLPALI_USE_MOCK=false.
Endpoints:
  GET  /healthz   -> {"embedding_dim": int, "model_id": str}
  POST /embed/query  JSON {"query": "..."}  -> {"vector": [...]}
  POST /embed/image  JSON {"image_b64": "..."} -> {"vector": [...]}

Run from repo root:
  PYTHONPATH=src COLPALI_USE_MOCK=false python scripts/embed_server.py
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Ensure src on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from v2.colpali_embedder import ColPaliEmbedder

app = FastAPI(title="ColPali Embed Server", version="0.1.0")
_embedder: Optional[ColPaliEmbedder] = None


class QueryBody(BaseModel):
    query: str = ""


class ImageBody(BaseModel):
    image_b64: str = ""


@app.on_event("startup")
def startup() -> None:
    global _embedder
    os.environ.setdefault("COLPALI_USE_MOCK", "false")
    if os.getenv("COLPALI_EMBED_URL"):
        os.environ.pop("COLPALI_EMBED_URL", None)
    _embedder = ColPaliEmbedder()


@app.get("/healthz")
def healthz():
    if _embedder is None:
        raise HTTPException(status_code=503, detail="embedder not ready")
    return {
        "status": "ok",
        "embedding_dim": _embedder.embedding_dim,
        "model_id": _embedder.model_id,
        "mock": _embedder.is_mock,
    }


@app.post("/embed/query")
def embed_query(body: QueryBody):
    if _embedder is None:
        raise HTTPException(status_code=503, detail="embedder not ready")
    if not body.query.strip():
        raise HTTPException(status_code=422, detail="query required")
    v = _embedder.embed_query(body.query)
    return {"vector": v.tolist(), "embedding_dim": len(v)}


@app.post("/embed/image")
def embed_image(body: ImageBody):
    if _embedder is None:
        raise HTTPException(status_code=503, detail="embedder not ready")
    if not body.image_b64:
        raise HTTPException(status_code=422, detail="image_b64 required")
    raw = base64.b64decode(body.image_b64)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(raw)
    tmp.close()
    try:
        v = _embedder.embed_image_path(tmp.name)
    finally:
        os.unlink(tmp.name)
    return {"vector": v.tolist(), "embedding_dim": len(v)}


if __name__ == "__main__":
    port = int(os.getenv("EMBED_SERVER_PORT", "8765"))
    uvicorn.run(app, host="0.0.0.0", port=port)
