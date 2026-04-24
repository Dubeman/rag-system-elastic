"""Run retrieval eval against the RAG API using Kaggle ground truth.

Usage:
    python eval/run_eval.py --pipeline v1
    python eval/run_eval.py --pipeline v1 --compare-passage-index rag_passages_v2
    python eval/run_eval.py --pipeline v1 --print-sample-hit-fields
    python eval/run_eval.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.v2.eval_retrieval import mean_reciprocal_rank, recall_at_k

QRELS_PATH = Path(__file__).parent / "cases" / "qrels.json"
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_API = "http://localhost:8000"
DEFAULT_K_VALUES = (5, 10)


def load_qrels(path: Path) -> tuple[list[dict], dict[str, Any] | None]:
    if not path.exists():
        print(f"[ERROR] qrels not found at {path}")
        print("Run first: python eval/build_qrels.py")
        sys.exit(1)
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return raw, None
    if isinstance(raw, dict) and "queries" in raw:
        return raw["queries"], raw.get("meta")
    print("[ERROR] qrels JSON must be a list or {\"meta\", \"queries\"}")
    sys.exit(1)


def evidence_id_from_result(r: dict, pipeline: str) -> str:
    eid = (r.get("evidence_id") or "").strip()
    if eid:
        return eid
    if pipeline == "v2":
        doc_id = r.get("doc_id", "")
        page_num = r.get("page_num", 0)
        return f"{doc_id}:{page_num}"
    filename = r.get("filename", "")
    doc_name = Path(filename).stem
    chunk_id = r.get("chunk_id", 0)
    return f"{doc_name}:{chunk_id}"


def doc_id_from_evidence(eid: str) -> str:
    if ":" in eid:
        return eid.split(":", 1)[0]
    return eid


def doc_recall_at_k(relevant_evidence: set[str], ranked_evidence: list[str], k: int) -> float:
    rel_docs = {doc_id_from_evidence(x) for x in relevant_evidence}
    if not rel_docs:
        return 0.0
    retrieved = {doc_id_from_evidence(x) for x in ranked_evidence[:k]}
    return len(rel_docs & retrieved) / len(rel_docs)


def doc_mean_reciprocal_rank(relevant_evidence: set[str], ranked_evidence: list[str]) -> float:
    rel_docs = {doc_id_from_evidence(x) for x in relevant_evidence}
    for i, eid in enumerate(ranked_evidence, start=1):
        if doc_id_from_evidence(eid) in rel_docs:
            return 1.0 / i
    return 0.0


def doc_precision_at_k(relevant_evidence: set[str], ranked_evidence: list[str], k: int) -> float:
    rel_docs = {doc_id_from_evidence(x) for x in relevant_evidence}
    if k <= 0 or not rel_docs:
        return 0.0
    hits = 0
    for eid in ranked_evidence[:k]:
        if doc_id_from_evidence(eid) in rel_docs:
            hits += 1
    return hits / k


def compute_precision_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / k


def query_rag(
    api_url: str,
    question: str,
    top_k: int,
    pipeline: str,
    *,
    elasticsearch_index: str | None = None,
    search_mode: str | None = None,
) -> tuple[list[str], list[dict]]:
    """Return (ranked_evidence_ids, raw_results) from /query."""
    payload: dict[str, Any] = {
        "question": question,
        "top_k": top_k,
        "generate_answer": False,
        "pipeline_version": pipeline,
    }
    if elasticsearch_index:
        payload["elasticsearch_index"] = elasticsearch_index
    if search_mode:
        payload["search_mode"] = search_mode

    resp = requests.post(f"{api_url}/query", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    ranked = [evidence_id_from_result(r, pipeline) for r in results]
    return ranked, results


def filter_qrels(
    qrels: list[dict],
    *,
    exclude_unanswerable: bool,
) -> tuple[list[dict], int]:
    excluded = 0
    out: list[dict] = []
    for q in qrels:
        rel = set(q.get("relevant", []))
        if exclude_unanswerable and len(rel) == 0:
            excluded += 1
            continue
        out.append(q)
    return out, excluded


def evaluate_queries(
    api_url: str,
    pipeline: str,
    qrels: list[dict],
    k_values: tuple[int, ...],
    *,
    elasticsearch_index: str | None,
    search_mode: str,
    smoke: bool,
) -> dict[str, Any]:
    if smoke:
        qrels = qrels[:5]
        print(f"[SMOKE] Running on first {len(qrels)} queries only")

    max_k = max(k_values)
    per_query: list[dict[str, Any]] = []

    if not qrels:
        print("[WARN] No queries to evaluate after filtering.")
        return {
            "per_query": [],
            "aggregate": {},
            "zero_recall_by_k": {str(k): 0 for k in k_values},
            "id_mismatch_confirmed": False,
            "num_queries_evaluated": 0,
        }

    print(f"\nQuerying pipeline={pipeline} @ {api_url} (max_k={max_k}, index={elasticsearch_index or 'default'})")
    print("-" * 60)

    for i, qrel in enumerate(qrels):
        qid = qrel["qid"]
        question = qrel["question"]
        relevant = set(qrel["relevant"])

        ranked, _raw = query_rag(
            api_url,
            question,
            max_k,
            pipeline,
            elasticsearch_index=elasticsearch_index,
            search_mode=search_mode,
        )

        row: dict[str, Any] = {
            "qid": qid,
            "question": question[:120],
            "relevant_count": len(relevant),
            "retrieved_count": len(ranked),
        }
        for k in k_values:
            rk = ranked[:k]
            row[f"evidence_recall@{k}"] = round(recall_at_k(relevant, ranked, k), 4)
            row[f"evidence_precision@{k}"] = round(compute_precision_at_k(relevant, ranked, k), 4)
            row[f"evidence_mrr@{k}"] = round(mean_reciprocal_rank(relevant, rk), 4)
            row[f"doc_recall@{k}"] = round(doc_recall_at_k(relevant, ranked, k), 4)
            row[f"doc_precision@{k}"] = round(doc_precision_at_k(relevant, ranked, k), 4)
            row[f"doc_mrr@{k}"] = round(doc_mean_reciprocal_rank(relevant, rk), 4)

        per_query.append(row)

        er = row.get(f"evidence_recall@{max_k}", 0.0)
        status = "✓" if er > 0 else "✗"
        print(f"  [{i+1:02d}/{len(qrels)}] {status} {qid} | evidence_r@{max_k}={er:.3f} | {question[:50]}...")

    # Aggregates
    def mean_key(key: str) -> float:
        return sum(float(row[key]) for row in per_query) / len(per_query)

    aggregate: dict[str, Any] = {}
    zero_by_k: dict[str, int] = {}
    for k in k_values:
        aggregate[f"evidence_recall@{k}"] = round(mean_key(f"evidence_recall@{k}"), 4)
        aggregate[f"evidence_precision@{k}"] = round(mean_key(f"evidence_precision@{k}"), 4)
        aggregate[f"evidence_mrr@{k}"] = round(mean_key(f"evidence_mrr@{k}"), 4)
        aggregate[f"doc_recall@{k}"] = round(mean_key(f"doc_recall@{k}"), 4)
        aggregate[f"doc_precision@{k}"] = round(mean_key(f"doc_precision@{k}"), 4)
        aggregate[f"doc_mrr@{k}"] = round(mean_key(f"doc_mrr@{k}"), 4)
        zero_by_k[str(k)] = sum(1 for row in per_query if row.get(f"evidence_recall@{k}", 0.0) == 0.0)

    print("-" * 60)
    for k in k_values:
        print(
            f"  @k={k}  evidence_recall={aggregate[f'evidence_recall@{k}']:.4f}  "
            f"doc_recall={aggregate[f'doc_recall@{k}']:.4f}  "
            f"zero_evidence_recall={zero_by_k[str(k)]}/{len(per_query)}"
        )

    id_mismatch_confirmed = False
    if elasticsearch_index is None and len(k_values) >= 1 and 5 in k_values:
        dr = aggregate.get("doc_recall@5", 0.0)
        er = aggregate.get("evidence_recall@5", 0.0)
        if dr - er > 0.15:
            id_mismatch_confirmed = True
            print(f"\nID mismatch confirmed: doc_recall@5 ({dr:.3f}) >> evidence_recall@5 ({er:.3f})")

    return {
        "per_query": per_query,
        "aggregate": aggregate,
        "zero_recall_by_k": zero_by_k,
        "id_mismatch_confirmed": id_mismatch_confirmed,
        "num_queries_evaluated": len(per_query),
    }


def build_results_payload(
    *,
    version: str,
    api_url: str,
    pipeline: str,
    k_values: tuple[int, ...],
    n_total: int,
    n_evaluated: int,
    n_excluded_unanswerable: int,
    eval_block: dict[str, Any],
    qrels_meta: dict[str, Any] | None,
    elasticsearch_index: str | None,
    search_mode: str,
    smoke: bool,
) -> dict[str, Any]:
    doc_recall = {f"@{k}": eval_block["aggregate"].get(f"doc_recall@{k}", 0.0) for k in k_values}
    doc_precision = {
        f"@{k}": eval_block["aggregate"].get(f"doc_precision@{k}", 0.0) for k in k_values
    }
    doc_mrr = {f"@{k}": eval_block["aggregate"].get(f"doc_mrr@{k}", 0.0) for k in k_values}
    evidence_recall = {f"@{k}": eval_block["aggregate"].get(f"evidence_recall@{k}", 0.0) for k in k_values}
    evidence_precision = {
        f"@{k}": eval_block["aggregate"].get(f"evidence_precision@{k}", 0.0) for k in k_values
    }
    evidence_mrr = {f"@{k}": eval_block["aggregate"].get(f"evidence_mrr@{k}", 0.0) for k in k_values}

    return {
        "version": version,
        "timestamp": datetime.utcnow().isoformat(),
        "k_values": list(k_values),
        "n_total": n_total,
        "n_evaluated": n_evaluated,
        "n_excluded_unanswerable": n_excluded_unanswerable,
        "api_url": api_url,
        "pipeline": pipeline,
        "search_mode": search_mode,
        "elasticsearch_index": elasticsearch_index,
        "smoke": smoke,
        "qrels_meta": qrels_meta,
        "doc_recall": doc_recall,
        "doc_precision": doc_precision,
        "doc_mrr": doc_mrr,
        "evidence_recall": evidence_recall,
        "evidence_precision": evidence_precision,
        "evidence_mrr": evidence_mrr,
        "zero_recall_count": eval_block["zero_recall_by_k"],
        "id_mismatch_confirmed": eval_block.get("id_mismatch_confirmed", False),
        "per_query": eval_block["per_query"],
    }


def print_sample_hit_fields(api_url: str, pipeline: str, qrels: list[dict]) -> None:
    if not qrels:
        print("[inspect] no qrels")
        return
    q = qrels[0]["question"]
    ranked, results = query_rag(api_url, q, top_k=1, pipeline=pipeline, search_mode="dense_bm25")
    if not results:
        print("[inspect] no hits; _source fields unavailable")
        return
    src = results[0].get("_source") or {}
    print(f"\n[inspect] sample question qid={qrels[0].get('qid')} top-1 _source keys: {sorted(src.keys())}")
    print(f"[inspect] derived evidence_id: {ranked[0] if ranked else '(none)'}")


def save_results(results: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    core = results.get("run", results)
    pv = core.get("pipeline", "v1")
    ver = str(core.get("version", pv)).replace("|", "-")
    out = RESULTS_DIR / f"eval_{ver}_{ts}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval eval against RAG API")
    parser.add_argument("--api", default=DEFAULT_API, help="RAG API base URL")
    parser.add_argument("--pipeline", choices=["v1", "v2"], default="v1")
    parser.add_argument(
        "--k-values",
        type=str,
        default=",".join(str(k) for k in DEFAULT_K_VALUES),
        help="Comma-separated k values (default: 5,10)",
    )
    parser.add_argument("--smoke", action="store_true", help="Run on first 5 queries only")
    parser.add_argument("--qrels", default=str(QRELS_PATH), help="Path to qrels JSON")
    parser.add_argument(
        "--exclude-unanswerable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude queries with empty relevant set (default: true)",
    )
    parser.add_argument("--search-mode", default="dense_bm25", help="v1 search_mode for /query")
    parser.add_argument(
        "--elasticsearch-index",
        default=None,
        help="Optional v1 Elasticsearch index (passage benchmark index)",
    )
    parser.add_argument(
        "--compare-passage-index",
        default=None,
        help="If set, also run eval against this index and include comparison block in output",
    )
    parser.add_argument(
        "--print-sample-hit-fields",
        action="store_true",
        help="Print _source keys for one sample hit and exit",
    )
    args = parser.parse_args()

    k_values = tuple(int(x.strip()) for x in args.k_values.split(",") if x.strip())
    if not k_values or any(k <= 0 for k in k_values):
        print("[ERROR] invalid --k-values")
        sys.exit(1)

    qrels_all, qrels_meta = load_qrels(Path(args.qrels))
    n_total = len(qrels_all)
    qrels, n_excluded = filter_qrels(qrels_all, exclude_unanswerable=args.exclude_unanswerable)
    print(
        f"Loaded {n_total} queries from {args.qrels}; "
        f"evaluating {len(qrels)} (excluded unanswerable: {n_excluded})"
    )

    if not qrels:
        print("[ERROR] No queries to evaluate after filtering.")
        sys.exit(1)

    if args.print_sample_hit_fields:
        print_sample_hit_fields(args.api, args.pipeline, qrels)
        return

    baseline = evaluate_queries(
        args.api,
        args.pipeline,
        qrels,
        k_values,
        elasticsearch_index=args.elasticsearch_index,
        search_mode=args.search_mode,
        smoke=args.smoke,
    )

    version = "passage_index" if args.elasticsearch_index else "v1"
    if args.pipeline == "v2":
        version = "v2"

    out_payload: dict[str, Any] = {
        "run": build_results_payload(
            version=version,
            api_url=args.api,
            pipeline=args.pipeline,
            k_values=k_values,
            n_total=n_total,
            n_evaluated=len(baseline["per_query"]),
            n_excluded_unanswerable=n_excluded,
            eval_block=baseline,
            qrels_meta=qrels_meta,
            elasticsearch_index=args.elasticsearch_index,
            search_mode=args.search_mode,
            smoke=args.smoke,
        )
    }

    if args.compare_passage_index and args.pipeline == "v1":
        passage = evaluate_queries(
            args.api,
            args.pipeline,
            qrels,
            k_values,
            elasticsearch_index=args.compare_passage_index,
            search_mode=args.search_mode,
            smoke=args.smoke,
        )
        out_payload["comparison"] = {
            "passage_index": args.compare_passage_index,
            "baseline": out_payload["run"],
            "passage": build_results_payload(
                version="passage_index",
                api_url=args.api,
                pipeline=args.pipeline,
                k_values=k_values,
                n_total=n_total,
                n_evaluated=len(passage["per_query"]),
                n_excluded_unanswerable=n_excluded,
                eval_block=passage,
                qrels_meta=qrels_meta,
                elasticsearch_index=args.compare_passage_index,
                search_mode=args.search_mode,
                smoke=args.smoke,
            ),
        }
        print("\n=== Side-by-side (baseline default index vs passage index) ===")
        for k in k_values:
            b = out_payload["run"]["evidence_recall"][f"@{k}"]
            p = out_payload["comparison"]["passage"]["evidence_recall"][f"@{k}"]
            bd = out_payload["run"]["doc_recall"][f"@{k}"]
            pd = out_payload["comparison"]["passage"]["doc_recall"][f"@{k}"]
            print(
                f"  @k={k}  evidence_recall  baseline={b:.4f}  passage={p:.4f}  |  "
                f"doc_recall  baseline={bd:.4f}  passage={pd:.4f}"
            )

    save_results(out_payload)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Cannot reach API — is the server running?")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP {e.response.status_code}: {e.response.text}")
        sys.exit(1)
