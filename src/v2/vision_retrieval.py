"""
Vision retrieval: ColPaliEmbedder (mock, HTTP, or local) + FAISS or Qdrant vector store.

Mean-pooled single vectors per page/query for ANN search; not full ColBERT MaxSim.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .colpali_embedder import ColPaliEmbedder
from .vector_store import build_vector_store

logger = logging.getLogger(__name__)


def _meta_path(data_dir: Path) -> Path:
    return Path(data_dir) / "embedding_meta.json"


def _load_meta(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
        return None


def _write_meta(path: Path, embedder: ColPaliEmbedder) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(embedder.config_signature(), indent=2),
        encoding="utf-8",
    )


def _meta_mismatch(
    stored: Optional[Dict[str, Any]], current: Dict[str, Any]
) -> bool:
    if stored is None:
        return False
    keys = ("model_id", "embedding_dim", "use_mock", "embed_url", "vector_backend")
    for k in keys:
        if stored.get(k) != current.get(k):
            return True
    return False


def _should_clear_index(
    stored: Optional[Dict[str, Any]],
    current: Dict[str, Any],
    row_count: int,
) -> bool:
    """Rebuild when config changed or legacy data without embedding_meta.json."""
    if row_count == 0:
        return False
    if stored is None:
        return True
    return _meta_mismatch(stored, current)


class VisionRetriever:
    """
    Page-level retriever: embed pages and queries, store vectors, search top-k.
    """

    def __init__(
        self,
        data_dir: Path,
        embedder: Optional[ColPaliEmbedder] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or ColPaliEmbedder()
        self._meta_file = _meta_path(self.data_dir)
        stored = _load_meta(self._meta_file)
        cur = self.embedder.config_signature()

        self._store = build_vector_store(self.data_dir, self.embedder.embedding_dim)
        self._store.load()

        if _should_clear_index(stored, cur, self._store.row_count):
            logger.warning(
                "Embedding config changed or missing embedding_meta.json with existing "
                "vectors — clearing index. Re-run v2 ingest."
            )
            self._store.clear()
            self._store.load()

        _write_meta(self._meta_file, self.embedder)

    def index_page_records(self, records: List[Dict[str, Any]]) -> int:
        added = 0
        for r in records:
            vec = self.embedder.embed_image_path(r["image_path"])
            meta = {
                "doc_id": r["doc_id"],
                "page_num": r["page_num"],
                "image_path": r["image_path"],
                "filename": r.get("filename", ""),
            }
            self._store.add(vec, meta)
            added += 1
        self._store.persist()
        _write_meta(self._meta_file, self.embedder)
        return added

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = self.embedder.embed_query(query)
        pairs = self._store.search(q, top_k=top_k)
        out: List[Dict[str, Any]] = []
        for score, meta in pairs:
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
                    "_score": score,
                    "_source": meta,
                    "search_type": "vision_colpali_mock"
                    if self.embedder.is_mock
                    else "vision_colpali",
                }
            )
        return out

    def search_mode_label(self) -> str:
        """API `search_mode` — reflects ColPali retrieval + vector backend (FAISS vs Qdrant)."""
        backend = os.getenv("VECTOR_BACKEND", "faiss").lower().strip()
        if backend == "qdrant":
            return "vision_colpali_qdrant"
        return "vision_colpali_faiss"
