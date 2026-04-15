# Recording v1 baselines

1. Run the stack with `PIPELINE_VERSION=v1` (default).
2. Call `POST /query` for each question in your validation set; note `total_ms` from `request_summary` logs or use `curl -w "%{time_total}"`.
3. Copy `baseline_metrics.example.json` to `baseline_metrics.json` and fill `query_latency_ms` and optional quality notes.
4. Keep the same validation set when comparing v2.
