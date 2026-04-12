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

## Repository layout

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

- `POST /query`: question -> answer (+ citations)
- `POST /ingest`: ingest/reindex documents
- `GET /healthz`: health check
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
