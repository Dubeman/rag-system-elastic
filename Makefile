SHELL := /bin/bash

KAGGLE_CSV ?= $(shell ls ../kaggle/outputs/pilot_labeled_full_*.csv 2>/dev/null | tail -1)
API_URL    ?= http://localhost:8000

.PHONY: env-core env-notebook env-train env-v2-gpu env-dev lock smoke-matrix smoke-infra \
        build-qrels seed-eval eval eval-smoke eval-v2

env-core:
	uv sync

env-notebook:
	uv sync --extra notebook

env-train:
	uv sync --extra train

env-v2-gpu:
	uv sync
	uv pip install -r requirements-colpali.txt

env-dev:
	uv sync --extra dev

lock:
	uv lock

smoke-matrix:
	bash scripts/validate_env_matrix.sh

smoke-infra:
	bash scripts/smoke_infra.sh

# --- Eval ---

seed-eval:
	uv run python eval/seed_eval_index.py

build-qrels:
	@if [ -z "$(KAGGLE_CSV)" ]; then echo "ERROR: no pilot_labeled_full_*.csv found in ../kaggle/outputs/"; exit 1; fi
	uv run python eval/build_qrels.py --csv $(KAGGLE_CSV)

# Requires API + Elasticsearch with corpus seeded (see eval/seed_eval_index.py and docker-compose.smoke.yml).
EVAL_RECALL_FLOOR ?= 0.6

eval:
	@test -f eval/cases/qrels.json || (echo "Missing eval/cases/qrels.json"; exit 1)
	uv run python eval/run_eval.py --api $(API_URL) --pipeline v1 --fail-under-evidence-recall-at-5 $(EVAL_RECALL_FLOOR)

eval-smoke:
	@test -f eval/cases/qrels.json || (echo "Missing eval/cases/qrels.json"; exit 1)
	uv run python eval/run_eval.py --api $(API_URL) --pipeline v1 --smoke --fail-under-evidence-recall-at-5 $(EVAL_RECALL_FLOOR)

eval-v2:
	@test -f eval/cases/qrels.json || (echo "Missing eval/cases/qrels.json"; exit 1)
	uv run python eval/run_eval.py --api $(API_URL) --pipeline v2
