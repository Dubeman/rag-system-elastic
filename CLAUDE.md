# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Environment setup (uv)
make env-core          # core runtime only
make env-dev           # + pytest, black, isort, flake8
make env-notebook      # + langgraph, langchain-core, langfuse, gradio
make env-v2-gpu        # + colpali-engine (GPU path)

# Tests
uv run pytest                                      # full suite with coverage
uv run pytest tests/unit/                         # unit tests only
uv run pytest tests/unit/test_retrieval.py -k "test_rrf"  # single test

# Eval (requires API running)
python eval/run_eval.py --pipeline v1 --smoke     # 5-query smoke check
python eval/run_eval.py --pipeline v1             # full eval, needs eval/cases/qrels.json
python scripts/eval_v2.py eval/fixtures/sample_eval.json  # v2 fixture eval

# Docker
docker compose up --build -d                      # full stack (ES + Ollama + API + UI)
docker compose -f docker-compose.smoke.yml up     # smoke stack
bash scripts/smoke_infra.sh                       # validate infra

# Linting
uv run black src tests && uv run isort src tests  # format
uv run flake8 src tests                           # lint

# Local dev servers (no Docker)
uv run uvicorn src.api.main:app --reload          # API on :8000
uv run streamlit run src/ui/app.py                # UI on :8501
python scripts/mock_vlm_server.py                 # mock VLM on :9999
```

## Architecture

Two parallel pipelines, toggled by `PIPELINE_VERSION=v1|v2` (env) or `"pipeline_version"` field in request body. Both share the same FastAPI app (`src/api/main.py`).

### v1 — Elasticsearch hybrid
`PDF → text chunks → Elasticsearch (BM25 + dense MiniLM + ELSER sparse) → RRF fusion → Ollama LLM`

- Indexing: `src/indexing/` (ElasticsearchClient, DocumentIndexer)
- Retrieval: `src/retrieval/` (HybridRetriever, CachedRetriever) — supports `bm25_only`, `dense_only`, `elser_only`, `dense_bm25`, `full_hybrid` modes
- Generation: `src/generation/` (AnswerGenerator + LLMClient → Ollama)
- Ingestion: `src/ingestion/` (PDF parser, enhanced chunker, Google Drive client)

### v2 — Vision-native (target architecture)
`PDF → page images (PyMuPDF) → ColPali embeddings → FAISS or Qdrant → VLM (RunPod/OpenAI-compat)`

- `src/v2/page_renderer.py` — PDF → PNG pages
- `src/v2/colpali_embedder.py` — embeds page images; supports mock, HTTP (`COLPALI_EMBED_URL`), or local `colpali-engine`
- `src/v2/vector_store.py` — FAISS or Qdrant backend, selected by `VECTOR_BACKEND`
- `src/v2/vlm_client.py` — HTTP VLM client (`RUNPOD_VLM_URL`), OpenAI-compat mode via `VLM_USE_OPENAI_COMPAT`
- `src/v2/pipeline_v2.py` — orchestrates ingest + retrieval + VLM answer

**North star gap:** v2 currently uses mean-pooled single vectors per page. The target is full ColPali late-interaction MaxSim (multi-vector per page). See `reports/OPTIMIZE_BACKLOG.md`.

### Key cross-cutting rules

**Re-embed rule:** `embedding_meta.json` under `V2_DATA_DIR` tracks `(model_id, dim, mock_flag, embed_url, backend)`. Changing any of these at startup clears the index and forces re-ingest. Do not skip this check.

**Observability:** Every request gets a `request_id` (UUID). Log `pipeline_version`, `request_id`, and `duration_ms` for each stage. Prometheus metrics via `src/observability/metrics.py`; Langfuse tracing via `src/observability/langfuse_utils.py`.

**Guardrails:** `src/guardrails/guardrails.py` — currently keyword-based. Known gap: should be replaced with LLM-as-judge calls.

### Eval harness
- `eval/run_eval.py` — main eval runner; hits live API; requires `eval/cases/qrels.json` (not committed; build with `eval/build_qrels.py`)
- Metrics: `evidence_recall@k`, `doc_recall@k`, `evidence_mrr@k`, `doc_mrr@k`, `precision@k`
- Results written to `eval/results/eval_<version>_<timestamp>.json`
- `src/v2/eval_retrieval.py` — standalone `recall_at_k`, `mean_reciprocal_rank` helpers (no API dependency)

## Zero-cost local development

All expensive components have free local alternatives. Use these during development:

```bash
COLPALI_USE_MOCK=true           # skip GPU embed server entirely
RUNPOD_VLM_URL=http://127.0.0.1:9999/  # use scripts/mock_vlm_server.py
VECTOR_BACKEND=faiss            # local disk, no Qdrant Cloud
LLM_SERVICE_URL=http://localhost:11434  # Ollama, free
```

Only spin up RunPod GPU for final validation runs (~1-2 hours, ~$1).

## Development workflow (PRD-based vertical slices)

Each piece of work is a **vertical slice**: retrieval → generation → eval, all wired end-to-end, even if thin. No horizontal layer work without an eval gate.

**Done = eval metric improves or holds.** A PR that drops `evidence_recall@5` does not merge.

Active PRD for the current slice lives below. Archive completed PRDs to `docs/prd/`.

---

## Current slice

**PRD:** [LLM-as-Judge Eval Pipeline](docs/prd/2026-05-05-llm-as-judge-eval.md)

**Goal:** `make eval` runs locally (Ollama, zero cost) and prints `recall@5` + MRR on 20–30 real queries.

**Done when:**
- `eval/cases/qrels.json` exists with 20–30 hand-labeled queries
- `make eval` produces a JSON results file in `eval/results/`
- GitHub Actions runs `--smoke` on every PR and gates on `evidence_recall@5`

**Out of scope:** ColPali/v2 changes, UI, cloud LLM calls, agents.
