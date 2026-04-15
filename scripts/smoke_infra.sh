#!/usr/bin/env bash
# Run from repo root: bash scripts/smoke_infra.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> pip install -r requirements.txt"
python3 -m pip install -q -r requirements.txt

echo "==> pytest v2 unit + pipeline smoke"
export PYTHONPATH=src
export COLPALI_USE_MOCK=true
export VECTOR_BACKEND=faiss
python3 -m pytest tests/unit/test_v2_eval.py tests/unit/test_v2_pipeline_smoke.py -q -o addopts=

echo "==> offline eval_v2"
python3 scripts/eval_v2.py eval/fixtures/sample_eval.json

echo "==> done. For full API /healthz see reports/SMOKE_RUNBOOK.md (Elasticsearch + uvicorn)."
