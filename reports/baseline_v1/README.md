# v1 baseline metrics

Record **Elasticsearch hybrid RAG + Ollama** baselines here before comparing v2.

## How to capture

1. Run the stack with `PIPELINE_VERSION=v1` (default).
2. Run a fixed validation query set (same questions each time).
3. Note p50 / p95 latency for `POST /query` and `POST /ingest` from logs (`request_summary` / `stage_timing`) or your load tool.
4. Save a snapshot as `baseline_metrics.json` (see `baseline_metrics.example.json`).

## Fields

- `date`: ISO date
- `git_ref`: commit SHA or tag (e.g. `v1.0-baseline`)
- `queries`: list of `{ "question": "...", "p50_ms": 0, "p95_ms": 0 }` or aggregate numbers
- `notes`: environment (Docker, hardware)

Replace example placeholders with measured values when you benchmark.
