"""
ColPali-style vision retrieval: embed page images and text queries, search with FAISS.

Uses deterministic mock embeddings unless you extend this module with real
``colpali-engine`` inference (same index API). Set ``COLPALI_USE_MOCK=false`` to
signal production embedding path once implemented.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


def _stable_unit_vector(key: bytes, dim: int = 128) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**31)
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    n = np.linalg.norm(v) + 1e-8
    return (v / n).astype(np.float32)


class VisionRetriever:
    """
    Page-level retriever: index (doc_id, page_num, image_path) with embeddings in FAISS.
    """

    def __init__(
        self,
        data_dir: Path,
        embedding_dim: int = 128,
        use_mock: Optional[bool] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.index_dir = self.data_dir / "faiss"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        if use_mock is None:
            use_mock = os.getenv("COLPALI_USE_MOCK", "true").lower() in (
                "1",
                "true",
                "yes",
            )
        self._use_mock = use_mock
        self._faiss_index: Optional[Any] = None
        self._meta: List[Dict[str, Any]] = []
        self._load_index()

    def _load_index(self) -> None:
        meta_path = self.index_dir / "manifest.json"
        index_path = self.index_dir / "index.faiss"
        if meta_path.exists():
            self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not _FAISS_AVAILABLE:
            logger.error("faiss-cpu is required for v2 vision retrieval")
            return
        if index_path.exists():
            self._faiss_index = faiss.read_index(str(index_path))
            logger.info(
                "Loaded FAISS index with %s vectors",
                self._faiss_index.ntotal,
            )
        else:
            self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)

    def _persist(self) -> None:
        meta_path = self.index_dir / "manifest.json"
        meta_path.write_text(json.dumps(self._meta, indent=2), encoding="utf-8")
        if _FAISS_AVAILABLE and self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(self.index_dir / "index.faiss"))

    def _embed_query_text(self, query: str) -> np.ndarray:
        # Replace with ColPali query tower when COLPALI_USE_MOCK=false and models are wired.
        _ = self._use_mock
        return _stable_unit_vector(("q:" + query).encode("utf-8"), self.embedding_dim)

    def _embed_image_path(self, image_path: str) -> np.ndarray:
        p = Path(image_path)
        raw = p.read_bytes() if p.exists() else b""
        return _stable_unit_vector(b"img:" + raw, self.embedding_dim)

    def index_page_records(self, records: List[Dict[str, Any]]) -> int:
        """Add page records (must include image_path, doc_id, page_num). Returns count added."""
        if not _FAISS_AVAILABLE or self._faiss_index is None:
            logger.warning("FAISS unavailable; skipping index updates")
            return 0

        added = 0
        for r in records:
            vec = self._embed_image_path(r["image_path"])
            x = vec.reshape(1, -1)
            self._faiss_index.add(x)
            self._meta.append(
                {
                    "doc_id": r["doc_id"],
                    "page_num": r["page_num"],
                    "image_path": r["image_path"],
                    "filename": r.get("filename", ""),
                }
            )
            added += 1
        self._persist()
        return added

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = self._embed_query_text(query)
        if not _FAISS_AVAILABLE or self._faiss_index is None:
            return []
        if self._faiss_index.ntotal == 0:
            return []

        k = min(top_k, self._faiss_index.ntotal)
        D, I = self._faiss_index.search(q.reshape(1, -1), k)
        out: List[Dict[str, Any]] = []
        for rank, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(self._meta):
                continue
            meta = self._meta[idx]
            out.append(
                {
                    "content": "",
                    "filename": meta.get("filename", ""),
                    "doc_id": meta["doc_id"],
                    "page_num": meta["page_num"],
                    "image_path": meta["image_path"],
                    "chunk_id": f"{meta['doc_id']}:{meta['page_num']}",
                    "file_url": "",
                    "modified_time": "",
                    "_score": float(D[0][rank]),
                    "_source": meta,
                    "search_type": "vision_colpali_mock"
                    if self._use_mock
                    else "vision_colpali",
                }
            )
        return out
