"""
ColPali-style embeddings: mock, HTTP microservice, or local GPU (colpali-engine).

For FAISS we use mean-pooled L2-normalized vectors (single vector per page/query).
Full multi-vector MaxSim is not stored in FAISS here; see colpali-engine docs for
score_multi_vector when you need late interaction.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)

MOCK_DIM = 128


def _stable_unit_vector(key: bytes, dim: int = MOCK_DIM) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**31)
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    n = np.linalg.norm(v) + 1e-8
    return (v / n).astype(np.float32)


def _pool_model_output_to_vector(t: Any) -> np.ndarray:
    """Turn model output (tensor or dict) into a single L2-normalized float32 vector."""
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("torch required for local ColPali embeddings") from e

    if torch.is_tensor(t):
        x = t.float()
    elif isinstance(t, dict):
        for k in ("last_hidden_state", "embeddings", "image_embeddings", "query_embeddings"):
            if k in t and t[k] is not None:
                x = t[k].float()
                break
        else:
            x = next(v for v in t.values() if torch.is_tensor(v)).float()
    elif isinstance(t, (list, tuple)) and len(t) > 0 and torch.is_tensor(t[0]):
        x = t[0].float()
    else:
        raise TypeError(f"Unexpected model output type: {type(t)}")

    if not hasattr(x, "dim"):
        raise TypeError(f"Unexpected model output type: {type(t)}")
    if x.dim() == 3:
        x = x.squeeze(0).mean(dim=0)
    elif x.dim() == 2:
        x = x.mean(dim=0)
    else:
        x = x.flatten()
    x = torch.nn.functional.normalize(x, dim=-1)
    return x.detach().cpu().numpy().astype(np.float32)


class ColPaliEmbedder:
    """
    Embedding backend selected by env:

    - COLPALI_USE_MOCK=true (default): deterministic mock vectors, dim=128.
    - COLPALI_EMBED_URL set: HTTP POST to embedding service (no local torch model).
    - Else: load ColPali locally (requires colpali-engine, GPU/MPS recommended).
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        use_mock: Optional[bool] = None,
        embed_url: Optional[str] = None,
    ) -> None:
        self.model_id = model_id or os.getenv(
            "COLPALI_MODEL_ID", "vidore/colpali-v1.2"
        )
        self.embed_url = (embed_url or os.getenv("COLPALI_EMBED_URL", "")).rstrip("/")

        if use_mock is None:
            use_mock = os.getenv("COLPALI_USE_MOCK", "true").lower() in (
                "1",
                "true",
                "yes",
            )
        self._use_mock = use_mock
        self._model = None
        self._processor = None
        self._device = None
        self._dim: int = MOCK_DIM

        if self._use_mock:
            self._dim = MOCK_DIM
            logger.info("ColPaliEmbedder: mock mode dim=%s", self._dim)
            return

        if self.embed_url:
            self._probe_http_dim()
            logger.info(
                "ColPaliEmbedder: HTTP backend %s dim=%s", self.embed_url, self._dim
            )
            return

        self._load_local_model()

    def _probe_http_dim(self) -> None:
        """Optional GET /health or first embed to learn dim; default MOCK_DIM until first embed."""
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.get(f"{self.embed_url}/healthz")
                if r.status_code == 200:
                    data = r.json()
                    self._dim = int(data.get("embedding_dim", data.get("dim", MOCK_DIM)))
                    return
        except Exception as e:
            logger.warning("Could not probe embed server dim: %s", e)
        self._dim = int(os.getenv("COLPALI_EMBEDDING_DIM", str(MOCK_DIM)))

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def is_mock(self) -> bool:
        return self._use_mock

    def config_signature(self) -> Dict[str, Any]:
        """Persisted to embedding_meta.json; mismatch triggers index rebuild."""
        sig: Dict[str, Any] = {
            "model_id": self.model_id,
            "embedding_dim": self._dim,
            "use_mock": self._use_mock,
            "embed_url": self.embed_url or "",
        }
        sig["vector_backend"] = os.getenv("VECTOR_BACKEND", "faiss").lower().strip()
        return sig

    def _load_local_model(self) -> None:
        try:
            import torch
            from colpali_engine.models import ColPali, ColPaliProcessor
            from colpali_engine.utils.torch_utils import get_torch_device
        except ImportError as e:
            raise RuntimeError(
                "Local ColPali requires: pip install colpali-engine (see requirements-colpali.txt). "
                "Or set COLPALI_EMBED_URL to a running embed server, or COLPALI_USE_MOCK=true."
            ) from e

        device = get_torch_device("auto")
        self._device = device
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        logger.info("Loading ColPali %s on %s ...", self.model_id, device)
        self._processor = ColPaliProcessor.from_pretrained(self.model_id)
        self._model = ColPali.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=None,
        )
        self._model = self._model.to(device)
        self._model.eval()

        # Probe dim with a tiny query
        with torch.no_grad():
            bq = self._processor.process_queries(["probe"]).to(self._device)
            qe = self._model(**bq)
            v = _pool_model_output_to_vector(qe)
        self._dim = int(v.shape[0])
        logger.info("ColPali local model loaded; embedding_dim=%s", self._dim)

    def embed_query(self, text: str) -> np.ndarray:
        if self._use_mock:
            return _stable_unit_vector(("q:" + text).encode("utf-8"), self._dim)

        if self.embed_url:
            return self._http_embed("query", text=text)

        import torch

        assert self._model is not None and self._processor is not None
        with torch.no_grad():
            bq = self._processor.process_queries([text]).to(self._device)
            out = self._model(**bq)
            return _pool_model_output_to_vector(out)

    def embed_image_path(self, image_path: str) -> np.ndarray:
        if self._use_mock:
            p = __import__("pathlib").Path(image_path)
            raw = p.read_bytes() if p.exists() else b""
            return _stable_unit_vector(b"img:" + raw, self._dim)

        if self.embed_url:
            return self._http_embed("image", image_path=image_path)

        import torch
        from PIL import Image

        assert self._model is not None and self._processor is not None
        img = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            bi = self._processor.process_images([img]).to(self._device)
            out = self._model(**bi)
            return _pool_model_output_to_vector(out)

    def _http_embed(self, kind: str, text: str = "", image_path: str = "") -> np.ndarray:
        assert self.embed_url
        payload: dict = {"model_id": self.model_id}
        if kind == "query":
            payload["query"] = text
            path = "/embed/query"
        else:
            raw = __import__("pathlib").Path(image_path).read_bytes()
            payload["image_b64"] = base64.b64encode(raw).decode("ascii")
            path = "/embed/image"

        url = f"{self.embed_url}{path}"
        with httpx.Client(timeout=120.0) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        vec = data.get("vector") or data.get("embedding")
        if not vec:
            raise ValueError(f"Embed server returned no vector: {data.keys()}")
        v = np.array(vec, dtype=np.float32)
        n = np.linalg.norm(v) + 1e-8
        v = (v / n).astype(np.float32)
        if v.shape[0] != self._dim:
            self._dim = int(v.shape[0])
        return v
