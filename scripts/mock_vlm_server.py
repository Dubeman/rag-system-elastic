#!/usr/bin/env python3
"""
Minimal HTTP server that mimics the custom JSON contract expected by
RunPodVLMClient (question + pages with image_b64).

Use for local "real inference" smoke without RunPod:
  export RUNPOD_VLM_URL=http://127.0.0.1:9999/
  python scripts/mock_vlm_server.py

Then POST /query with pipeline_version v2 — llm_response.status will be "ok".
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI(title="Mock VLM", version="0.1.0")


class VlmPayload(BaseModel):
    question: str = ""
    pages: List[Dict[str, Any]] = []
    model: str = ""


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "mock_vlm"}


@app.post("/")
async def invoke_custom(request: Request):
    """Accept raw JSON (question + pages) from RunPodVLMClient."""
    body = await request.json()
    question = body.get("question", "")
    pages = body.get("pages") or []
    return {
        "answer": f"[mock-vlm] Received question with {len(pages)} page image(s). Preview: {question[:80]!r}",
        "citations": [
            {"doc_id": p.get("doc_id"), "page_num": p.get("page_num")}
            for p in pages[:3]
        ],
        "status": "ok",
    }


@app.post("/v1/chat/completions")
async def openai_compat(request: Request):
    """Optional OpenAI-style body; not used when VLM_USE_OPENAI_COMPAT=false."""
    _ = await request.json()
    return {
        "choices": [
            {
                "message": {
                    "content": "[mock-vlm] OpenAI-compat stub response.",
                }
            }
        ]
    }


if __name__ == "__main__":
    port = int(os.getenv("MOCK_VLM_PORT", "9999"))
    uvicorn.run(app, host="0.0.0.0", port=port)
