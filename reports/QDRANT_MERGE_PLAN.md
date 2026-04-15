# Qdrant + vision v2 on `main`

This document records how **v2** page vectors use **FAISS** (default, on disk under `V2_DATA_DIR`) or **Qdrant** (HTTP), while **v1** stays on **Elasticsearch** (BM25, dense, ELSER, RRF). It is not a replacement of ELSER or v1 ingestion.

## Branch strategy

- Integration branch: **`feat/vision-v2-qdrant-merge`** — merge of **`feat/vision-v2`** into **`main`**, resolving conflicts in `README.md`, `requirements.txt`, and `src/api/main.py`.
- **Default:** `PIPELINE_VERSION=v1` so existing deployments behave as before until v2 is opted in.

## v1 vs v2

| | v1 | v2 |
|---|----|----|
| Storage | Elasticsearch indices | `VECTOR_BACKEND=faiss` or `qdrant` under `V2_DATA_DIR` + `embedding_meta.json` |
| Ingest | Text chunks + ES indexing | PDF → page images → ColPali-class embed → vector upsert |
| Query | Hybrid retriever + Ollama LLM | Vision retrieval + `RUNPOD_VLM_URL` (VLM) |

## Running Qdrant

1. **Docker Compose:** a **`qdrant`** service is defined in `docker-compose.yml` (port **6333**, volume `qdrant_data`). Set **`VECTOR_BACKEND=qdrant`** and **`QDRANT_URL=http://qdrant:6333`** on the **`api`** service (see commented env in compose file).
2. **Local:** run the official image or binary on `localhost:6333` and set `QDRANT_URL=http://localhost:6333`.
3. **Qdrant Cloud:** HTTPS URL + `QDRANT_API_KEY` as in `env.example`.

## Re-embed / index invalidation

Changing **`COLPALI_MODEL_ID`**, **`COLPALI_EMBED_URL`**, mock vs real, or **`VECTOR_BACKEND`** invalidates the v2 index per `embedding_meta.json` — **re-run v2 ingest**.

## Merge conflict resolution (reference)

- **`src/api/main.py`:** Combined **Prometheus** helpers from `src/observability/metrics.py` (`setup_prometheus_instrumentation`, `rag_*` stage metrics) with **v2** routing (`PIPELINE_VERSION`, `VisionPipelineV2`), **`src/api/observability.py`** request/stage logging, and a separate histogram **`rag_pipeline_http_duration_seconds`** (avoids clashing with `prometheus-fastapi-instrumentator`’s `http_request_duration_seconds`).
- **`requirements.txt`:** `prometheus-fastapi-instrumentator` plus v2 deps (`pymupdf`, `faiss-cpu`, `qdrant-client`).
- **`README.md`:** v1 observability docs + v2 endpoints and metrics notes.

## Rollback

- Revert the merge PR or set **`PIPELINE_VERSION=v1`** and disable v2-only env. Qdrant data is disposable in dev; use snapshots/backups in prod.
