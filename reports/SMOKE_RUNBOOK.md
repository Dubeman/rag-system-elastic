# Simple infra smoke runbook

Follow this order: environment, v2 mock E2E, one real inference path, baselines, then defer optimization.

## 1. Install and environment

```bash
cd /path/to/rag-system-elastic
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
cp -n env.example .env
```

### Elasticsearch for `/healthz` (API startup pings ES)

Minimal local ES (no ELSER setup):

```bash
docker compose -f docker-compose.smoke.yml up -d elasticsearch
# wait until healthy
export ELASTICSEARCH_URL=http://localhost:9200
```

Point `LLM_SERVICE_URL` at Ollama if you have it, or ignore v1 LLM for v2-only smoke.

### Start API

```bash
export PYTHONPATH=src
export COLPALI_USE_MOCK=true
export V2_DATA_DIR=data/v2
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Check:

```bash
curl -s http://localhost:8000/healthz | python3 -m json.tool
```

Expect `vision_v2`: `ready` when startup succeeds.

## 2. v2 mock end-to-end (no GPU)

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source":"sample","sample_text":"Smoke test document about Docker and APIs.","pipeline_version":"v2"}' | python3 -m json.tool

curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this document about?","top_k":3,"generate_answer":true,"pipeline_version":"v2"}' | python3 -m json.tool
```

Expect `llm_response.status` = `stub` unless `RUNPOD_VLM_URL` is set. Expect `timings_ms` in the response.

Automated (no ES): `PYTHONPATH=src pytest tests/unit/test_v2_pipeline_smoke.py -q`

## 3. First real inference (pick one)

### Option A — Mock VLM HTTP (no RunPod bill)

Terminal 1:

```bash
python scripts/mock_vlm_server.py
```

Terminal 2 — API env:

```bash
export RUNPOD_VLM_URL=http://127.0.0.1:9999/
# trailing slash matches POST to that base URL
```

Restart API, repeat v2 query. Expect `llm_response.status` = `ok`.

### Option B — Real ColPali embed server (GPU)

On GPU host:

```bash
pip install -r requirements.txt -r requirements-colpali.txt
export COLPALI_USE_MOCK=false
export EMBED_SERVER_PORT=8765
PYTHONPATH=src python scripts/embed_server.py
```

On API host:

```bash
export COLPALI_USE_MOCK=false
export COLPALI_EMBED_URL=http://<gpu-host>:8765
```

Re-run v2 ingest (embedding config change clears index).

## 4. Baselines and eval

- v1: see [BASELINE_STEPS.md](baseline_v1/BASELINE_STEPS.md)
- Offline eval: `python scripts/eval_v2.py eval/fixtures/sample_eval.json`
- Fill [v1_vs_v2_baseline_template.md](v1_vs_v2_baseline_template.md)

## 5. Optimization backlog

See [OPTIMIZE_BACKLOG.md](OPTIMIZE_BACKLOG.md) — do after steps 1–4.
