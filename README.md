# RAG System with Elasticsearch + Open-Source LLMs

Hybrid Retrieval-Augmented Generation pipeline using **Elasticsearch (ELSER sparse)**, **BM25**, and **dense embeddings**, fused with **RRF**, then served via **FastAPI** with a **Streamlit** UI.

## What it does

Given a document collection (PDFs from Google Drive), the system:

1. Ingests PDFs and chunks text
2. Builds retrieval indexes (sparse + dense + keyword)
3. Retrieves relevant passages using a configurable fusion strategy (RRF)
4. Generates answers with an open-source LLM (via Ollama)
5. Returns answers with citations (retrieved sources)

## Architecture

![System Architecture](Hexaware%20internship.drawio.png)

## Key technical ideas

- **Hybrid retrieval**: `ELSER (sparse)` + `Dense embeddings` + `BM25`
- **Fusion**: **Reciprocal Rank Fusion (RRF)** over multiple retrieval rankings
- **Generation**: Open-source LLM integration with guardrails / safety checks (see `src/guardrails/`)
- **API-first design**: FastAPI exposes `ingest` and `query` endpoints; Streamlit is a UI layer

## Tech stack

- Elasticsearch 8.x (ELSER + hybrid search)
- `sentence-transformers/all-MiniLM-L6-v2` (dense embeddings)
- FastAPI (API)
- Streamlit (UI)
- Ollama (local LLM serving)
- Docker Compose (reproducible setup)

## Versions and migration (v1 vs v2)

Active development for **vision-native document intelligence** happens on branch **`feat/vision-v2`**.

| Tag | Meaning |
|-----|---------|
| `v1.0-baseline` | Text-first Elasticsearch hybrid RAG + Ollama (comparison baseline) |
| `v2.0-vision` | Vision pipeline scaffolding (ColPali-style + FAISS + RunPod VLM hook); re-tag after benchmarks |

Set **`PIPELINE_VERSION`** to `v1` (default) or `v2`. See [AGENTS.md](AGENTS.md) for model stack and agent conventions.

### v1 vs v2 (high level)

| | v1 (baseline) | v2 (vision-native) |
|---|----------------|-------------------|
| Ingestion | PDF text + chunking | PDF page rasterization (PyMuPDF) |
| Retrieval | BM25 + dense + ELSER + RRF | ColPali-style embeddings (mean-pooled for FAISS/Qdrant; see `src/v2/colpali_embedder.py`) |
| Generation | Ollama LLM | Phi-3.5-vision / SmolVLM via `RUNPOD_VLM_URL` (stub if unset) |
| API flag | `PIPELINE_VERSION=v1` or `pipeline_version` in JSON body | same, use `v2` |

Endpoints: `POST /ingest`, `POST /query` (pass `"pipeline_version": "v2"`), `GET /metrics` (Prometheus), `GET /healthz` (includes `vision_v2`).

### Vision v2: where things run

| Component | Role | Configuration |
|-----------|------|-----------------|
| **ColPali embeddings** | Query + page image vectors (pretrained; optional LoRA later) | `COLPALI_USE_MOCK=true` (dev default), or `COLPALI_USE_MOCK=false` + local GPU + `pip install -r requirements-colpali.txt`, or **`COLPALI_EMBED_URL`** pointing at [`scripts/embed_server.py`](scripts/embed_server.py) on RunPod |
| **Vector index** | Stores vectors + page metadata | **`VECTOR_BACKEND=faiss`** (files under `V2_DATA_DIR`, gitignored) or **`VECTOR_BACKEND=qdrant`** + `QDRANT_URL` / `QDRANT_COLLECTION` |
| **VLM** | Answer + citations from top page images | **`RUNPOD_VLM_URL`**; set **`VLM_USE_OPENAI_COMPAT=true`** if the endpoint is OpenAI-style `chat/completions` |

**Re-embed rule:** `embedding_meta.json` under `V2_DATA_DIR` tracks model id, dim, mock flag, embed URL, and vector backend. If you change any of these, the index is cleared — **re-run v2 ingest**.

**Eval:** Offline JSON example in [`eval/fixtures/sample_eval.json`](eval/fixtures/sample_eval.json). Run `python scripts/eval_v2.py eval/fixtures/sample_eval.json` or pass `--api-url http://localhost:8000` for live recall (requires ingested v2 corpus).

**Baselines:** See [`reports/baseline_v1/BASELINE_STEPS.md`](reports/baseline_v1/BASELINE_STEPS.md) and copy `baseline_metrics.example.json` to record v1 numbers before comparing v2.

## Repository layout

- `src/v2/`: vision-native ingestion, FAISS retrieval, RunPod VLM client, eval helpers
- `src/ingestion/`: PDF ingestion + chunking
- `src/indexing/`: indexing/index rebuild logic for Elasticsearch
- `src/retrieval/`: hybrid retrieval + fusion strategies
- `src/generation/`: LLM integration and answer generation
- `src/api/`: FastAPI endpoints and request validation
- `src/ui/`: Streamlit application
- `tests/`: unit + integration tests

## Quick start (Docker Compose)

Prerequisites:
- Docker + Docker Compose
- Enough RAM/disk for Elasticsearch + model downloads

1. Clone
   ```bash
   git clone https://github.com/Dubeman/rag-system-elastic.git
   cd rag-system-elastic
   ```

2. Configure environment (optional)
   ```bash
   cp env.example .env
   ```

3. Start the system
   ```bash
   docker compose up --build -d
   ```

4. Ingest documents
   - This README uses **public Google Drive folder access** (no Google API credentials required):
     - Share the folder so it is viewable by link
     - Provide the folder id to the `POST /ingest` call

5. Query
   - API: `http://localhost:8000/query`
   - UI: `http://localhost:8501`

## API endpoints

- `POST /query`: question -> answer (+ citations); optional `"pipeline_version": "v2"`
- `POST /ingest`: ingest/reindex documents; optional `"pipeline_version": "v2"`
- `GET /healthz`: health check (includes Elasticsearch, LLM, vision v2)
- `GET /metrics`: Prometheus metrics (latency histogram `http_request_duration_seconds`)

## Performance note

- Target end-to-end latency: `<= 3 seconds`.
- Current PoC latency is higher (`~1 minute`) due to model downloads and runtime conditions; optimization opportunities are documented in the repo.

## Development

- Code quality: pre-commit hooks (`.pre-commit-config.yaml`)
- Tests:
  ```bash
  pytest tests/ -v --cov=src/
  ```

## License

MIT
