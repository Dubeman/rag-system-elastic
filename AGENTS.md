# Agent context — Document Intelligence v2

## Mission

Migrate from legacy text-first RAG (v1) to **vision-native** retrieval and answering (v2): page images, ColPali-style embeddings, Phi-3.5-vision (or SmolVLM) via RunPod, optional PEFT LoRA.

## Branches and tags

- **Development:** `feat/vision-v2`
- **Baseline tag:** `v1.0-baseline` — last v1-focused state for comparison
- **Release target:** `v2.0-vision` — when v2 metrics meet README criteria

## Pipeline versions

- **`v1`:** Elasticsearch hybrid (BM25 + dense + ELSER + RRF) + Ollama LLM; text chunking.
- **`v2`:** PDF page images → ColPali-style retrieval (FAISS) → VLM on RunPod.

Set `PIPELINE_VERSION=v1` or `v2`. Default remains `v1` for backward compatibility.

## Agent rules

- Prefer thin vertical slices; instrument before optimizing.
- Do not remove v1 paths without benchmark justification.
- Log `request_id`, `pipeline_version`, and per-stage `duration_ms`.
