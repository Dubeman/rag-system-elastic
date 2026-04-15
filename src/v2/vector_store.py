"""Pluggable vector storage: FAISS on disk (default) or Qdrant (HTTP)."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class VectorStore(ABC):
    """Storage for page vectors + metadata."""

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def add(self, vector: np.ndarray, meta: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Return (score, meta) sorted by score descending."""

    @abstractmethod
    def persist(self) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @property
    @abstractmethod
    def row_count(self) -> int:
        pass


class FAISSVectorStore(VectorStore):
    """FAISS IndexFlatIP + JSON manifest of metadata rows."""

    def __init__(self, index_dir: Path, embedding_dim: int) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self._faiss_index: Optional[Any] = None
        self._meta_rows: List[Dict[str, Any]] = []
        self._manifest_path = self.index_dir / "manifest.json"
        self._index_path = self.index_dir / "index.faiss"

    def load(self) -> None:
        if not _FAISS_AVAILABLE:
            logger.error("faiss-cpu not installed")
            return
        if self._manifest_path.exists():
            raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                self._meta_rows = raw
            else:
                self._meta_rows = raw.get("rows", [])
        if self._index_path.exists():
            self._faiss_index = faiss.read_index(str(self._index_path))
        else:
            self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)

    def add(self, vector: np.ndarray, meta: Dict[str, Any]) -> None:
        if not _FAISS_AVAILABLE or self._faiss_index is None:
            return
        x = vector.reshape(1, -1).astype(np.float32)
        self._faiss_index.add(x)
        self._meta_rows.append(meta)

    def search(
        self, query_vector: np.ndarray, top_k: int
    ) -> List[Tuple[float, Dict[str, Any]]]:
        if not _FAISS_AVAILABLE or self._faiss_index is None:
            return []
        if self._faiss_index.ntotal == 0:
            return []
        k = min(top_k, self._faiss_index.ntotal)
        D, I = self._faiss_index.search(
            query_vector.reshape(1, -1).astype(np.float32), k
        )
        out: List[Tuple[float, Dict[str, Any]]] = []
        for rank, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(self._meta_rows):
                continue
            out.append((float(D[0][rank]), dict(self._meta_rows[idx])))
        return out

    def persist(self) -> None:
        manifest = {"version": 2, "rows": self._meta_rows}
        self._manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if _FAISS_AVAILABLE and self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(self._index_path))

    def clear(self) -> None:
        self._meta_rows = []
        if _FAISS_AVAILABLE:
            self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        for p in (self._manifest_path, self._index_path):
            if p.exists():
                p.unlink()

    @property
    def row_count(self) -> int:
        if self._faiss_index is None:
            return 0
        return int(self._faiss_index.ntotal)


class QdrantVectorStore(VectorStore):
    """Qdrant collection; vectors live in Qdrant server."""

    def __init__(
        self,
        embedding_dim: int,
        collection_name: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams
        except ImportError as e:
            raise RuntimeError("pip install qdrant-client") from e

        self._embedding_dim = embedding_dim
        self._collection = collection_name or os.getenv(
            "QDRANT_COLLECTION", "v2_pages"
        )
        self._url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self._api_key = api_key or os.getenv("QDRANT_API_KEY", "")
        self._client = QdrantClient(url=self._url, api_key=self._api_key or None)
        self._PointStruct = PointStruct
        self._VectorParams = VectorParams
        self._Distance = Distance
        self._next_id = 0

        if not self._qdrant_collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=embedding_dim, distance=Distance.COSINE
                ),
            )

    def _qdrant_collection_exists(self, name: str) -> bool:
        try:
            return bool(self._client.collection_exists(name))
        except Exception:
            names = {c.name for c in self._client.get_collections().collections}
            return name in names

    def load(self) -> None:
        info = self._client.get_collection(self._collection)
        self._next_id = int(info.points_count)

    def add(self, vector: np.ndarray, meta: Dict[str, Any]) -> None:
        pid = self._next_id
        self._next_id += 1
        self._client.upsert(
            collection_name=self._collection,
            points=[
                self._PointStruct(
                    id=pid,
                    vector=vector.astype(np.float32).tolist(),
                    payload=meta,
                )
            ],
        )

    def search(
        self, query_vector: np.ndarray, top_k: int
    ) -> List[Tuple[float, Dict[str, Any]]]:
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector.astype(np.float32).tolist(),
            limit=top_k,
        )
        out: List[Tuple[float, Dict[str, Any]]] = []
        for h in hits:
            pl = dict(h.payload or {})
            out.append((float(h.score), pl))
        return out

    def persist(self) -> None:
        pass

    def clear(self) -> None:
        if self._qdrant_collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=self._VectorParams(
                size=self._embedding_dim, distance=self._Distance.COSINE
            ),
        )
        self._next_id = 0

    @property
    def row_count(self) -> int:
        return int(self._client.get_collection(self._collection).points_count)


def build_vector_store(data_dir: Path, embedding_dim: int) -> VectorStore:
    backend = os.getenv("VECTOR_BACKEND", "faiss").lower().strip()
    if backend == "qdrant":
        return QdrantVectorStore(embedding_dim=embedding_dim)
    return FAISSVectorStore(Path(data_dir) / "faiss", embedding_dim=embedding_dim)
