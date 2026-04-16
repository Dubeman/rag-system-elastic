"""Unit tests for v2 retrieval eval helpers (no live services)."""

from v2.eval_retrieval import (
    aggregate_recall,
    mean_reciprocal_rank,
    page_key,
    recall_at_k,
)


def test_page_key():
    assert page_key("doc1", 3) == "doc1:3"


def test_recall_at_k():
    rel = {"a:0", "b:1"}
    ranked = ["x:9", "y:8", "a:0"]
    assert recall_at_k(rel, ranked, k=2) == 0.0
    ranked2 = ["a:0", "b:1"]
    assert recall_at_k(rel, ranked2, k=2) == 1.0
    # Standard Recall@k: denominator is |relevant|, not min(|relevant|, k).
    three_rel = {"a:0", "b:1", "c:2"}
    assert recall_at_k(three_rel, ["a:0", "b:1"], k=2) == 2.0 / 3.0


def test_mrr():
    assert mean_reciprocal_rank({"a"}, ["b", "a", "c"]) == 0.5
    assert mean_reciprocal_rank({"z"}, ["a", "b"]) == 0.0


def test_aggregate_recall():
    qrels = [
        {"qid": "1", "relevant": ["d:0"]},
        {"qid": "2", "relevant": ["d:1", "d:2"]},
    ]
    runs = [
        {"qid": "1", "ranked": ["d:0", "x:1"]},
        {"qid": "2", "ranked": ["d:9", "d:1"]},
    ]
    assert aggregate_recall(qrels, runs, k=1) > 0.0
