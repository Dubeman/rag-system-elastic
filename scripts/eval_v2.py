#!/usr/bin/env python3
"""
Offline retrieval evaluation from JSON (Recall@k / aggregate).

Example:
  python scripts/eval_v2.py eval/fixtures/sample_eval.json

With API (optional, requires running server + ingested v2 index):
  python scripts/eval_v2.py --api-url http://localhost:8000 eval/fixtures/sample_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from v2.eval_retrieval import aggregate_recall, page_key  # noqa: E402


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def eval_offline(data: Dict[str, Any]) -> None:
    k = int(data.get("k", 5))
    score = aggregate_recall(data["qrels"], data["runs"], k=k)
    print(f"aggregate_recall@{k}: {score:.4f}")


def eval_api(api_url: str, data: Dict[str, Any]) -> None:
    import httpx

    k = int(data.get("k", 5))
    qrels = data["qrels"]
    scores = []
    for row in qrels:
        qid = row["qid"]
        question = row.get("question", f"eval:{qid}")
        r = httpx.post(
            f"{api_url.rstrip('/')}/query",
            json={
                "question": question,
                "top_k": k,
                "generate_answer": False,
                "pipeline_version": "v2",
            },
            timeout=120.0,
        )
        r.raise_for_status()
        out = r.json()
        ranked: List[str] = []
        for hit in out.get("results", []):
            ranked.append(
                page_key(str(hit.get("doc_id", "")), int(hit.get("page_num", 0)))
            )
        rel = set(row["relevant"])
        from v2.eval_retrieval import recall_at_k

        scores.append(recall_at_k(rel, ranked, k))
    print(f"mean recall@{k} over {len(scores)} queries: {sum(scores)/len(scores):.4f}")


def main() -> int:
    p = argparse.ArgumentParser(description="v2 retrieval eval")
    p.add_argument("fixture", type=Path, help="JSON file with qrels/runs or qrels+questions")
    p.add_argument("--api-url", default="", help="If set, run live queries against FastAPI")
    args = p.parse_args()
    data = load_json(args.fixture)
    if args.api_url:
        eval_api(args.api_url, data)
    else:
        eval_offline(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
