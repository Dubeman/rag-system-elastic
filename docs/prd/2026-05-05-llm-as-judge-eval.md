# PRD: LLM-as-Judge Eval Pipeline (Ollama, zero cost)

**Status:** active

## Goal
A `make eval` command that runs retrieval eval locally using Ollama as the relevance judge and prints `recall@5`, `MRR`, and per-query pass/fail — with no external API calls and no GPU required.

## Metric
`evidence_recall@5 >= 0.6` on a manually curated set of 20–30 real queries against the ingested corpus. This becomes the CI baseline gate.

## Done when
- `make eval` runs end-to-end against a local Docker stack
- Produces a JSON results file in `eval/results/`
- A GitHub Actions workflow runs `--smoke` (5 queries) on every PR and fails if `evidence_recall@5` drops below the baseline

## Scope
- **In:** qrels fixture (20–30 queries), Ollama-based relevance grader, `make eval` target, GitHub Actions CI step
- **Out:** ColPali / v2 changes, UI changes, cloud LLM calls, agent patterns

## Implementation notes
- Relevance grader: single Ollama prompt per (query, chunk) pair — "Is this passage relevant to the question? Answer yes or no." — replaces the keyword overlap check in `src/generation/generator.py:check_context_relevance`
- Qrels: hand-label 20–30 queries using documents already ingested; store in `eval/cases/qrels.json`
- CI: `.github/workflows/eval.yml` — spins up `docker-compose.smoke.yml`, runs `python eval/run_eval.py --smoke`, asserts threshold
