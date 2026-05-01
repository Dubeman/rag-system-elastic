SHELL := /bin/bash

.PHONY: env-core env-notebook env-train env-v2-gpu env-dev lock smoke-matrix smoke-infra

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
