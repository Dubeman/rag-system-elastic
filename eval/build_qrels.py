"""Convert Kaggle labeled CSV into qrels format for retrieval eval.

Usage:
    python eval/build_qrels.py
    python eval/build_qrels.py --csv path/to/custom.csv --out eval/cases/qrels.json
    python eval/build_qrels.py --audit-only
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

KAGGLE_OUTPUTS = Path(__file__).parent.parent.parent / "kaggle" / "outputs"
DEFAULT_OUT = Path(__file__).parent / "cases" / "qrels.json"
RELEVANCE_THRESHOLD = 2  # label >= 2 counts as relevant

# Candidate column names for optional fields (audit picks first present)
PASSAGE_TEXT_CANDIDATES = ("passage_text", "passage", "text", "context")
ANSWER_CANDIDATES = ("answer", "gold_answer", "target_answer", "label_answer")


def find_latest_csv(outputs_dir: Path) -> Path:
    pattern = str(outputs_dir / "pilot_labeled_full_*.csv")
    matches = sorted(glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No pilot_labeled_full_*.csv found in {outputs_dir}\n"
            "Run build_analysis_dataset.py in the kaggle project first."
        )
    return Path(matches[-1])


def _non_null_rate(series: Any) -> float:
    if series is None or len(series) == 0:
        return 0.0
    return float(series.notna().sum()) / float(len(series))


def _pick_column(df: "pd.DataFrame", candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def audit_csv(csv_path: Path, threshold: int) -> dict[str, Any]:
    """Print and return audit summary + decision gate for eval pipeline."""
    if pd is None:
        raise ImportError("pandas required: pip install pandas")

    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    print("\n=== CSV audit ===")
    print(f"Path: {csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns ({len(cols)}): {cols}")

    required = {"query_idx", "query_text", "doc_name", "passage_id", "relevance_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    passage_text_col = _pick_column(df, PASSAGE_TEXT_CANDIDATES)
    answer_col = _pick_column(df, ANSWER_CANDIDATES)

    print("\n--- Optional columns ---")
    print(f"passage_text column: {passage_text_col or '(none)'}")
    if passage_text_col:
        rate = _non_null_rate(df[passage_text_col])
        nonempty = (df[passage_text_col].fillna("").astype(str).str.len() > 0).mean()
        print(f"  non-null rate: {rate:.3f}")
        print(f"  non-empty string rate: {nonempty:.3f}")
    print(f"gold answer column: {answer_col or '(none)'}")
    if answer_col:
        rate = _non_null_rate(df[answer_col])
        nonempty = (df[answer_col].fillna("").astype(str).str.len() > 0).mean()
        print(f"  non-null rate: {rate:.3f}")
        print(f"  non-empty string rate: {nonempty:.3f}")

    print("\n--- Sample rows (first 3) ---")
    for i, row in df.head(3).iterrows():
        print(f"  row_index={i}")
        for c in cols:
            val = row[c]
            s = str(val)
            if len(s) > 200:
                s = s[:200] + "..."
            print(f"    {c}: {s}")

    n_unanswerable = 0
    for query_idx in sorted(df["query_idx"].unique()):
        rows = df[df["query_idx"] == query_idx]
        rel = (rows["relevance_label"] >= threshold).sum()
        if rel == 0:
            n_unanswerable += 1

    print("\n--- Relevance (threshold=%s) ---" % threshold)
    print(f"Unique queries: {df['query_idx'].nunique()}")
    print(f"Queries with zero relevant rows (unanswerable): {n_unanswerable}")

    # Decision gate
    passage_ok = bool(
        passage_text_col
        and _non_null_rate(df[passage_text_col]) > 0.5
        and (df[passage_text_col].fillna("").astype(str).str.len() > 0).mean() > 0.5
    )
    answer_ok = bool(
        answer_col
        and _non_null_rate(df[answer_col]) > 0.1
        and (df[answer_col].fillna("").astype(str).str.len() > 0).mean() > 0.1
    )

    print("\n=== Decision gate ===")
    if passage_ok:
        print("passage_text present -> passage-aligned index path ENABLED")
        phase3 = "full"
    else:
        print(
            "passage_text not found or too sparse -> passage-aligned index DISABLED; "
            "doc-level-only eval path"
        )
        phase3 = "skip"
        if answer_ok:
            print(
                "passage_text absent + gold answer present -> "
                "answer-span fallback path ENABLED (Phase 3B)"
            )
            phase3 = "fallback_only"
        else:
            print(
                "passage_text absent and no usable gold answer column -> "
                "evidence-level claims disabled beyond doc-level"
            )

    return {
        "csv_path": str(csv_path.resolve()),
        "columns": cols,
        "passage_text_column": passage_text_col,
        "answer_column": answer_col,
        "passage_text_ok": passage_ok,
        "answer_ok": answer_ok,
        "n_rows": int(len(df)),
        "n_unique_queries": int(df["query_idx"].nunique()),
        "n_unanswerable_queries": int(n_unanswerable),
        "relevance_threshold": threshold,
        "phase3_passage_index": phase3,
    }


def build_qrels(csv_path: Path, threshold: int = RELEVANCE_THRESHOLD) -> list[dict]:
    """Read labeled CSV and return qrels list (query entries only)."""
    if pd is None:
        raise ImportError("pandas required: pip install pandas")

    df = pd.read_csv(csv_path)

    required_cols = {"query_idx", "query_text", "doc_name", "passage_id", "relevance_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    qrels: list[dict] = []
    for query_idx in sorted(df["query_idx"].unique()):
        rows = df[df["query_idx"] == query_idx]
        question = rows.iloc[0]["query_text"]

        relevant = [
            f"{row['doc_name']}:{int(row['passage_id'])}"
            for _, row in rows.iterrows()
            if row["relevance_label"] >= threshold
        ]

        qrels.append(
            {
                "qid": f"q{int(query_idx)}",
                "question": question,
                "relevant": relevant,
            }
        )

    return qrels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build qrels from Kaggle labeled CSV")
    parser.add_argument("--csv", type=str, default=None, help="Path to labeled CSV")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output JSON path")
    parser.add_argument(
        "--threshold",
        type=int,
        default=RELEVANCE_THRESHOLD,
        help="Min relevance_label to count as relevant (default: 2)",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Print audit + decision gate and exit without writing qrels",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else find_latest_csv(KAGGLE_OUTPUTS)
    print(f"Loading: {csv_path}")

    audit = audit_csv(csv_path, args.threshold)
    if args.audit_only:
        return

    qrels = build_qrels(csv_path, threshold=args.threshold)
    total_relevant = sum(len(q["relevant"]) for q in qrels)

    payload = {
        "meta": {
            **audit,
            "queries_count": len(qrels),
            "total_relevant_evidence": total_relevant,
        },
        "queries": qrels,
    }

    print(f"\nQueries: {len(qrels)}")
    print(f"Total relevant passages: {total_relevant}")
    print(f"Avg relevant per query: {total_relevant / len(qrels):.1f}")
    print(
        f"Unanswerable queries (zero relevant rows at threshold): "
        f"{audit['n_unanswerable_queries']} (see meta; eval may exclude these)"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved qrels (wrapped with meta) → {out_path}")


if __name__ == "__main__":
    main()
