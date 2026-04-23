"""Unit tests for eval helpers (no live API)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_run_eval():
    path = REPO_ROOT / "eval" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("run_eval_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_doc_recall_at_k_multi_doc():
    re = _load_run_eval()
    relevant = {"paperA:0", "paperB:1"}
    ranked = ["paperB:9", "paperC:2", "paperA:5"]
    # top-2 retrieves paperB and paperC -> one of two rel docs
    assert re.doc_recall_at_k(relevant, ranked, 2) == pytest.approx(0.5)
    # top-3 retrieves both rel docs
    assert re.doc_recall_at_k(relevant, ranked, 3) == pytest.approx(1.0)


def test_doc_mrr():
    re = _load_run_eval()
    relevant = {"x:1", "y:2"}
    ranked = ["z:0", "y:9", "x:1"]
    assert re.doc_mean_reciprocal_rank(relevant, ranked) == pytest.approx(1.0 / 2)


def test_load_qrels_wrapped(tmp_path):
    re = _load_run_eval()
    p = tmp_path / "q.json"
    p.write_text(
        json.dumps(
            {
                "meta": {"foo": 1},
                "queries": [{"qid": "q1", "question": "hi", "relevant": ["a:0"]}],
            }
        )
    )
    qs, meta = re.load_qrels(p)
    assert len(qs) == 1
    assert meta == {"foo": 1}


def test_load_qrels_plain_list(tmp_path):
    re = _load_run_eval()
    p = tmp_path / "q2.json"
    p.write_text(json.dumps([{"qid": "q1", "question": "hi", "relevant": []}]))
    qs, meta = re.load_qrels(p)
    assert len(qs) == 1
    assert meta is None


def test_filter_unanswerable():
    re = _load_run_eval()
    qrels = [
        {"qid": "q1", "question": "a", "relevant": []},
        {"qid": "q2", "question": "b", "relevant": ["d:1"]},
    ]
    out, ex = re.filter_qrels(qrels, exclude_unanswerable=True)
    assert ex == 1
    assert len(out) == 1 and out[0]["qid"] == "q2"
