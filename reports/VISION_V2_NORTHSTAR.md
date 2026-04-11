# Vision v2 — north star (authoritative)

This document is the **target architecture** for vision-native document intelligence. Agents and contributors should align implementation **toward** this; the repo may still be an **intermediate** scaffold (see [Current repo vs north star](#current-repo-vs-north-star)).

## Mission

**PDF → page images → ColPali-class embeddings → vector search → VLM answers** (with citations grounded in retrieved pages). **Text-first Elasticsearch hybrid RAG (v1)** remains a comparison baseline, not the destination for v2 product work.

## Stack (target)

| Layer | Role | Preferred tools |
|-------|------|-----------------|
| **Page rasterization** | PDF bytes → PNGs per page | PyMuPDF (`src/v2/page_renderer.py`) |
| **Vision retrieval encoder** | Image + query → vectors (ideally multi-vector / late interaction long-term) | `colpali-engine`; HF `vidore/colpali-v1.2` or `vidore/colqwen2-v1.0` |
| **Vector store** | **Not** Elasticsearch for ColPali vectors | **FAISS** (dev) / **Qdrant** (cloud or self-hosted); multi-vector MaxSim is a later upgrade |
| **VLM** | Answer + citations from top-k page images | SmolVLM (easy start) or Phi-3.5-vision; **HTTP** via `RUNPOD_VLM_URL` or OpenAI-compatible (`VLM_USE_OPENAI_COMPAT`) |
| **API** | Orchestration only | FastAPI (`src/api/main.py`) |

## Deployment shape (typical)

- **CPU**: API + optional Qdrant client.
- **GPU A**: Embedding server (`scripts/embed_server.py`); API sets **`COLPALI_EMBED_URL`**.
- **GPU B**: VLM (e.g. vLLM on RunPod); API sets **`RUNPOD_VLM_URL`**.

## Non-goals (for v2 retrieval)

- **Do not** use Elasticsearch as the primary store for ColPali page vectors (v1 ES is separate).
- **Do not** assume ELSER or BM25 text hybrid is required for v2 core retrieval.

## Model access (reference)

- **ColPali / ColQwen2:** Hugging Face; `pip install colpali-engine` for local embedding path.
- **Phi-3.5-vision:** gated — HF account + license acceptance + `huggingface-cli login`.
- **SmolVLM:** ungated, good default for first GPU inference experiments.

## Current repo vs north star

| North star | Today in repo |
|------------|----------------|
| Multi-vector ColPali + MaxSim in Qdrant | **Mean-pooled single vector** per page for ANN; see `reports/OPTIMIZE_BACKLOG.md` |
| Lean v2-only stack | **Monolith**: v1 + ES + v2 in one process |
| PDF upload API | **sample** + **google_drive** ingest; no multipart PDF upload yet |
| Production hardening | Backlog (auth on embed/VLM, rate limits) |

**Phased priority:** (1) reliable embed + search + VLM HTTP, (2) Qdrant in prod paths, (3) multi-vector / MaxSim, (4) LoRA and fine-tuning.

## Related

- `AGENTS.md` — branches, tags, agent conventions.
- `.cursor/rules/vision-v2-northstar.mdc` — Cursor rule that enforces this direction in sessions.
