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

Stack diagrams (Mermaid: Compose topology, ingest/query flows, observability, v2 note): [reports/RAG_STACK_DIAGRAM.md](reports/RAG_STACK_DIAGRAM.md).

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
- Prometheus + Grafana (optional; metrics and dashboards when using Compose)

## Versions and migration (v1 vs v2)

Active development for **vision-native document intelligence** happens on branch **`feat/vision-v2`**.

| Tag | Meaning |
|-----|---------|
| `v1.0-baseline` | Text-first Elasticsearch hybrid RAG + Ollama (comparison baseline) |
| `v2.0-vision` | Vision pipeline scaffolding (ColPali-style + FAISS + RunPod VLM hook); re-tag after benchmarks |

Set **`PIPELINE_VERSION`** to `v1` (default) or `v2`. See [AGENTS.md](AGENTS.md) for model stack and agent conventions.

**Vision v2 + Qdrant (Compose, env, v1 vs v2):** [reports/QDRANT_MERGE_PLAN.md](reports/QDRANT_MERGE_PLAN.md).

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
- `GET /healthz`: health check (Elasticsearch, LLM, vision v2)
- `GET /metrics`: Prometheus scrape endpoint (when `METRICS_ENABLED=true`)

## Observability (Prometheus and Grafana)

With Docker Compose, the stack can run **Prometheus** (metrics storage) and **Grafana** (dashboards) on the same network as the API. These are **local dev instances**, not Grafana Cloud.

| Service      | URL                       | Purpose |
|-------------|---------------------------|---------|
| RAG API     | `http://localhost:8000`   | App endpoints and **`GET /metrics`** (Prometheus text format). |
| Prometheus  | `http://localhost:9090`   | Scrapes the API every **15s** (`docker/prometheus/prometheus.yml`). |
| Grafana     | `http://localhost:3000`   | Dashboards; default login **`admin` / `admin`** (from Compose). Password reset email is **not** configured locally—use those credentials. |

**Disable metrics:** set `METRICS_ENABLED=false` on the API (see `env.example`). When disabled, custom RAG counters/histograms are not recorded and HTTP instrumentation (including `/metrics`) is not registered.

### What Prometheus scrapes

Prometheus uses one job, **`rag-api`**, targeting **`api:8000`** with **`metrics_path: /metrics`**. That endpoint exposes:

1. **HTTP metrics** (via `prometheus-fastapi-instrumentator`), including request counts and durations (e.g. `http_requests_total`, `http_request_duration_seconds`) with **handler** labels. The `/metrics` route is excluded from request-duration instrumentation. Other defaults (e.g. request/response size, in-progress requests) depend on the instrumentator version.

2. **Custom RAG metrics** (`src/observability/metrics.py`), with **bounded labels** only (no per-question text):
   - `rag_stage_duration_seconds` (histogram) — `stage`: `retrieve`, `generate`, or `ingest`; `search_mode`: valid retrieval mode or `unknown` / `none` (ingest).
   - `rag_llm_generation_total` (counter) — `status`: `success`, `error`, or `skipped` (when `generate_answer` is false).
   - `rag_cache_total` (counter) — `result`: `hit` or `miss`.
   - `rag_ingest_chunks_total` / `rag_ingest_errors_total` (counters) — chunks indexed vs indexing errors on ingest.

**Request correlation:** the API sets **`X-Request-ID`** on responses (client-supplied or generated). It is not a Prometheus label.

**Not scraped by default:** Elasticsearch, Ollama, and Streamlit (add exporters or instrumentation separately if needed).

### Grafana dashboard

Grafana is provisioned with a **Prometheus** datasource (`uid: prometheus`) pointing at `http://prometheus:9090`, and a starter dashboard **RAG API Overview** (`docker/grafana/dashboards/rag-overview.json`, UID `rag-api-overview`). Panels include:

- **HTTP request rate** — overall throughput from HTTP metrics.
- **HTTP request duration p95 (by handler)** — latency by route/handler.
- **RAG stage duration p95** — retrieve vs generate vs ingest from `rag_stage_duration_seconds`.
- **LLM generation outcomes** — rate of success, error, and skipped generations.
- **Retrieval cache hit ratio** — from `rag_cache_total` over a 5-minute window.

After `docker compose up`, open Grafana, sign in, and use **Dashboards → RAG API Overview**. Call `/query` and `/ingest` so series appear (scrapes every 15 seconds).

**Additional metrics:** ingest/query also populate `rag_pipeline_http_duration_seconds` (labels `method`, `path`, `pipeline_version`) alongside instrumentator HTTP metrics and the custom RAG metrics above.

## Performance note

- Target end-to-end latency: `<= 3 seconds`.
- Current PoC latency is higher (`~1 minute`) due to model downloads and runtime conditions; optimization opportunities are documented in the repo.

## Development

- Code quality: pre-commit hooks (`.pre-commit-config.yaml`)
- **Smoke (simple infra):** [reports/SMOKE_RUNBOOK.md](reports/SMOKE_RUNBOOK.md) — install, v2 mock E2E, optional mock VLM / real embed server; run `bash scripts/smoke_infra.sh` for pytest + offline eval.
- Tests:
  ```bash
  pytest tests/ -v --cov=src/
  ```

## License

MIT
