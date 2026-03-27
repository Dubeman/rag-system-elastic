# Agent context — Document Intelligence v2

## Mission

Migrate from legacy text-first RAG (v1) to **vision-native** retrieval and answering (v2): page images, ColPali-style embeddings, Phi-3.5-vision (or SmolVLM) via RunPod, optional PEFT LoRA.

## Branches and tags

- **Development:** `feat/vision-v2`
- **Baseline tag:** `v1.0-baseline` — last v1-focused state for comparison
- **Release target:** `v2.0-vision` — when v2 metrics meet README criteria

## Pipeline versions

- **`v1`:** Elasticsearch hybrid (BM25 + dense + ELSER + RRF) + Ollama LLM; text chunking.
- **`v2`:** PDF page images → ColPaliEmbedder (mock, HTTP embed server, or local `colpali-engine`) → FAISS or Qdrant → VLM HTTP client.

Set `PIPELINE_VERSION=v1` or `v2`. Default remains `v1` for backward compatibility.

## Deploy layout (typical)

- **CPU API** (FastAPI): orchestrates ingest, calls embed + vector search + VLM.
- **GPU RunPod A — embedding server:** run `scripts/embed_server.py` with `COLPALI_USE_MOCK=false`; set **`COLPALI_EMBED_URL`** on the API host.
- **GPU RunPod B — VLM:** pretrained Phi-3.5-vision / SmolVLM; set **`RUNPOD_VLM_URL`** (and **`VLM_USE_OPENAI_COMPAT`** if the server speaks OpenAI chat).
- **Vector state:** default **FAISS on disk** under `V2_DATA_DIR` (cheap). Optional **Qdrant** (`VECTOR_BACKEND=qdrant`) on a small always-on VM or Qdrant Cloud.

Re-index when **`COLPALI_MODEL_ID`**, embedding mode, or **`VECTOR_BACKEND`** changes (`embedding_meta.json` enforces this).

## Agent rules

- Prefer thin vertical slices; instrument before optimizing.
- Do not remove v1 paths without benchmark justification.
- Log `request_id`, `pipeline_version`, and per-stage `duration_ms`.
