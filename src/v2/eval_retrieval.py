"""Lightweight retrieval evaluation helpers (DocVQA-style page ids)."""

from __future__ import annotations

from typing import Iterable, List, Set


def page_key(doc_id: str, page_num: int) -> str:
    return f"{doc_id}:{page_num}"


def recall_at_k(relevant: Set[str], ranked: List[str], k: int) -> float:
    """Standard Recall@k: |relevant ∩ top-k| / |relevant|."""
    if not relevant or k <= 0:
        return 0.0
    top_k = set(ranked[:k])
    return len(top_k & relevant) / len(relevant)


def mean_reciprocal_rank(relevant: Set[str], ranked: List[str]) -> float:
    for i, rid in enumerate(ranked, start=1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def aggregate_recall(
    qrels: Iterable[dict],
    runs: Iterable[dict],
    k: int,
) -> float:
    """
    qrels: iterable of { "qid": str, "relevant": [ "doc:page", ... ] }
    runs: iterable of { "qid": str, "ranked": [ "doc:page", ... ] }
    """
    rel_map = {r["qid"]: set(r["relevant"]) for r in qrels}
    run_map = {r["qid"]: r["ranked"] for r in runs}
    scores = []
    for qid, rel in rel_map.items():
        ranked = run_map.get(qid, [])
        scores.append(recall_at_k(rel, ranked, k))
    return sum(scores) / len(scores) if scores else 0.0
