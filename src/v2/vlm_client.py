"""Vision-language answering via RunPod (Phi-3.5-vision / SmolVLM-compatible HTTP API)."""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


def _parse_answer_from_json(data: Dict[str, Any]) -> str:
    """Support custom JSON, OpenAI chat.completions, or plain {answer:}."""
    if data.get("answer"):
        return str(data["answer"])
    choices = data.get("choices")
    if choices and isinstance(choices, list):
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if content:
            return str(content)
    return ""


class RunPodVLMClient:
    """
    Calls a custom JSON endpoint or OpenAI-compatible chat/completions on RunPod.

    Env:
      RUNPOD_VLM_URL — POST URL (custom body or .../v1/chat/completions)
      RUNPOD_API_KEY — optional Bearer token
      VLM_USE_OPENAI_COMPAT — if true, send OpenAI-style vision chat payload
      VLM_MODEL_NAME — model id for OpenAI-compatible calls

    If RUNPOD_VLM_URL is unset, returns a stub response for local development.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("RUNPOD_VLM_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY", "")
        self.timeout_s = timeout_s
        self.use_openai_compat = os.getenv(
            "VLM_USE_OPENAI_COMPAT", "false"
        ).lower() in ("1", "true", "yes")

    def is_configured(self) -> bool:
        return bool(self.base_url)

    def generate_with_pages(
        self,
        question: str,
        retrieved_pages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build answer with citations from retrieved page metadata.

        ``retrieved_pages`` items should include doc_id, page_num, image_path, score.
        """
        if not self.is_configured():
            return self._stub_response(question, retrieved_pages)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.use_openai_compat:
            body = self._build_openai_vision_body(question, retrieved_pages)
        else:
            images_payload: List[Dict[str, Any]] = []
            for p in retrieved_pages[:8]:
                path = p.get("image_path", "")
                if path and Path(path).is_file():
                    raw = Path(path).read_bytes()
                    images_payload.append(
                        {
                            "doc_id": p.get("doc_id"),
                            "page_num": p.get("page_num"),
                            "image_b64": base64.b64encode(raw).decode("ascii"),
                        }
                    )
            body = {
                "question": question,
                "pages": images_payload,
                "model": os.getenv(
                    "VLM_MODEL_NAME", "microsoft/Phi-3.5-vision-instruct"
                ),
            }

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                r = client.post(self.base_url, json=body, headers=headers)
                r.raise_for_status()
                data = r.json()
            answer = _parse_answer_from_json(data)
            if not answer.strip():
                answer = str(data)
            citations = data.get("citations") or [
                {"doc_id": p.get("doc_id"), "page_num": p.get("page_num")}
                for p in retrieved_pages[:3]
            ]
            return {
                "answer": answer,
                "citations": citations,
                "status": "ok",
                "raw": data,
            }
        except Exception as e:
            logger.exception("VLM request failed: %s", e)
            return {
                "answer": f"VLM request failed: {e}",
                "citations": [],
                "status": "error",
                "error": str(e),
            }

    def _build_openai_vision_body(
        self, question: str, retrieved_pages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """OpenAI-style multimodal user message (first page only if many providers limit)."""
        content: List[Dict[str, Any]] = [{"type": "text", "text": question}]
        for p in retrieved_pages[:4]:
            path = p.get("image_path", "")
            if path and Path(path).is_file():
                raw = Path(path).read_bytes()
                b64 = base64.b64encode(raw).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                        },
                    }
                )
        return {
            "model": os.getenv(
                "VLM_MODEL_NAME", "microsoft/Phi-3.5-vision-instruct"
            ),
            "messages": [{"role": "user", "content": content}],
            "max_tokens": int(os.getenv("VLM_MAX_TOKENS", "1024")),
        }

    def _stub_response(
        self, question: str, retrieved_pages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        cites = [
            {"doc_id": p.get("doc_id"), "page_num": p.get("page_num")}
            for p in retrieved_pages[:3]
        ]
        return {
            "answer": (
                "[v2 stub] Set RUNPOD_VLM_URL for Phi-3.5-vision / SmolVLM. "
                "Use VLM_USE_OPENAI_COMPAT=true for OpenAI-style chat/completions. "
                f"Question: {question[:120]!r}. Retrieved {len(retrieved_pages)} page(s)."
            ),
            "citations": cites,
            "status": "stub",
        }
