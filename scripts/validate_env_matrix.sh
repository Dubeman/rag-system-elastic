#!/usr/bin/env bash
# Validate baseline contract across v1/v2 API and unit smoke.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
API_URL="${API_URL:-http://127.0.0.1:8000}"

echo "==> Running lightweight unit smoke (v2)"
export PYTHONPATH=src
export COLPALI_USE_MOCK=true
export VECTOR_BACKEND=faiss
"$PYTHON_BIN" -m pytest tests/unit/test_v2_pipeline_smoke.py tests/unit/test_v2_eval.py -q -o addopts=

echo "==> API health check (${API_URL}/healthz)"
curl -fsS "${API_URL}/healthz" >/dev/null

echo "==> v1 query smoke"
curl -fsS -X POST "${API_URL}/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this project?","top_k":3,"generate_answer":false,"pipeline_version":"v1"}' \
  >/dev/null

echo "==> v2 ingest/query smoke (mock mode)"
curl -fsS -X POST "${API_URL}/ingest" \
  -H "Content-Type: application/json" \
  -d '{"source":"sample","sample_text":"UV migration smoke document.","pipeline_version":"v2"}' \
  >/dev/null

curl -fsS -X POST "${API_URL}/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the smoke document mention?","top_k":3,"generate_answer":false,"pipeline_version":"v2"}' \
  >/dev/null

echo "==> Baseline parity contract passed"
