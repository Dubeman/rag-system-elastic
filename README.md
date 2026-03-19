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
