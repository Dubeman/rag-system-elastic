# Optimization backlog (after simple infra + first real inference)

Do not start these until v2 mock E2E works and at least one real inference path (VLM or ColPali embed) is validated.

1. **VRAM / split services** — Keep ColPali embed server and VLM on separate GPU endpoints if a single GPU OOMs.
2. **Qdrant** — Set `VECTOR_BACKEND=qdrant` and run Qdrant for durable remote index (`src/v2/vector_store.py`).
3. **Multi-vector MaxSim** — Replace mean-pooled FAISS vectors with full ColPali late interaction (`score_multi_vector` or external engine). Larger refactor.
4. **LoRA** — `scripts/train_lora.py` with labeled JSONL; serve base + adapter on your GPU stack.
5. **Production hardening** — Auth on embed/VLM HTTP, rate limits, index backup/restore runbooks.
