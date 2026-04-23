"""Run retrieval eval against the RAG API using Kaggle ground truth.

Usage:
    # Full eval, v1 pipeline, recall@5
    python eval/run_eval.py --pipeline v1 --k 5

    # Full eval, v2 pipeline, recall@10
    python eval/run_eval.py --pipeline v2 --k 10

    # Smoke test only (first 5 queries, fast)
    python eval/run_eval.py --smoke --k 5

    # Custom API URL
    python eval/run_eval.py --api http://localhost:8000 --pipeline v1 --k 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# Adjust path so we can import eval_retrieval from src/v2
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.v2.eval_retrieval import recall_at_k, mean_reciprocal_rank, aggregate_recall

QRELS_PATH = Path(__file__).parent / "cases" / "qrels.json"
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_API = "http://localhost:8000"


def load_qrels(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[ERROR] qrels not found at {path}")
        print("Run first: python eval/build_qrels.py")
        sys.exit(1)
    return json.loads(path.read_text())


def query_rag(api_url: str, question: str, top_k: int, pipeline: str) -> list[str]:
    """Hit /query and return ranked chunk IDs in order."""
    payload = {
        "question": question,
        "top_k": top_k,
        "generate_answer": False,
        "pipeline_version": pipeline,
    }
    try:
        resp = requests.post(f"{api_url}/query", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Cannot reach API at {api_url} — is Docker running?")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] API returned {e.response.status_code}: {e.response.text}")
        sys.exit(1)

    results = data.get("results", [])

    # Build canonical chunk IDs: {doc_id}:{page_num} for v2, {filename_stem}:{chunk_id} for v1
    chunk_ids = []
    for r in results:
        if pipeline == "v2":
            doc_id = r.get("doc_id", "")
            page_num = r.get("page_num", 0)
            chunk_ids.append(f"{doc_id}:{page_num}")
        else:
            # v1: filename is doc_name (e.g. "2311.16502v3.pdf"), chunk_id is passage index
            filename = r.get("filename", "")
            doc_name = Path(filename).stem  # strip .pdf extension
            chunk_id = r.get("chunk_id", 0)
            chunk_ids.append(f"{doc_name}:{chunk_id}")

    return chunk_ids


def compute_precision_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / k


def run_eval(
    api_url: str,
    pipeline: str,
    k: int,
    qrels: list[dict],
    smoke: bool = False,
) -> dict:
    if smoke:
        qrels = qrels[:5]
        print(f"[SMOKE] Running on first {len(qrels)} queries only")

    runs = []
    per_query = []

    print(f"\nQuerying {pipeline} pipeline @ {api_url} (top_k={k})")
    print("-" * 60)

    for i, qrel in enumerate(qrels):
        qid = qrel["qid"]
        question = qrel["question"]
        relevant = set(qrel["relevant"])

        ranked = query_rag(api_url, question, k, pipeline)
        runs.append({"qid": qid, "ranked": ranked})

        r_at_k = recall_at_k(relevant, ranked, k)
        mrr = mean_reciprocal_rank(relevant, ranked)
        p_at_k = compute_precision_at_k(relevant, ranked, k)

        per_query.append({
            "qid": qid,
            "question": question[:80],
            "relevant_count": len(relevant),
            "retrieved_count": len(ranked),
            f"recall@{k}": round(r_at_k, 4),
            f"precision@{k}": round(p_at_k, 4),
            "mrr": round(mrr, 4),
        })

        status = "✓" if r_at_k > 0 else "✗"
        print(f"  [{i+1:02d}/{len(qrels)}] {status} {qid} | "
              f"recall@{k}={r_at_k:.3f} mrr={mrr:.3f} | {question[:50]}...")

    # Aggregate
    mean_recall = sum(q[f"recall@{k}"] for q in per_query) / len(per_query)
    mean_mrr = sum(q["mrr"] for q in per_query) / len(per_query)
    mean_precision = sum(q[f"precision@{k}"] for q in per_query) / len(per_query)
    zero_recall = sum(1 for q in per_query if q[f"recall@{k}"] == 0)

    print("-" * 60)
    print(f"  mean recall@{k}:    {mean_recall:.4f}")
    print(f"  mean precision@{k}: {mean_precision:.4f}")
    print(f"  mean MRR:          {mean_mrr:.4f}")
    print(f"  queries with 0 recall: {zero_recall}/{len(per_query)}")

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline": pipeline,
        "k": k,
        "api_url": api_url,
        "smoke": smoke,
        "num_queries": len(qrels),
        "aggregate": {
            f"recall@{k}": round(mean_recall, 4),
            f"precision@{k}": round(mean_precision, 4),
            "mrr": round(mean_mrr, 4),
            "zero_recall_queries": zero_recall,
        },
        "per_query": per_query,
    }


def save_results(results: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"{results['pipeline']}_{ts}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Run retrieval eval against RAG API")
    parser.add_argument("--api", default=DEFAULT_API, help="RAG API base URL")
    parser.add_argument("--pipeline", choices=["v1", "v2"], default="v1")
    parser.add_argument("--k", type=int, default=5, help="Recall@k cutoff")
    parser.add_argument("--smoke", action="store_true", help="Run on first 5 queries only")
    parser.add_argument("--qrels", default=str(QRELS_PATH), help="Path to qrels JSON")
    args = parser.parse_args()

    qrels = load_qrels(Path(args.qrels))
    print(f"Loaded {len(qrels)} queries from {args.qrels}")

    results = run_eval(
        api_url=args.api,
        pipeline=args.pipeline,
        k=args.k,
        qrels=qrels,
        smoke=args.smoke,
    )

    save_results(results)


if __name__ == "__main__":
    main()
