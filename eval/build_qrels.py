"""Convert Kaggle labeled CSV into qrels format for retrieval eval.

Usage:
    python eval/build_qrels.py
    python eval/build_qrels.py --csv path/to/custom.csv --out eval/cases/qrels.json
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path


KAGGLE_OUTPUTS = Path(__file__).parent.parent.parent / "kaggle" / "outputs"
DEFAULT_OUT = Path(__file__).parent / "cases" / "qrels.json"
RELEVANCE_THRESHOLD = 2  # label >= 2 counts as relevant


def find_latest_csv(outputs_dir: Path) -> Path:
    pattern = str(outputs_dir / "pilot_labeled_full_*.csv")
    matches = sorted(glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No pilot_labeled_full_*.csv found in {outputs_dir}\n"
            "Run build_analysis_dataset.py in the kaggle project first."
        )
    return Path(matches[-1])  # latest by filename timestamp


def build_qrels(csv_path: Path, threshold: int = RELEVANCE_THRESHOLD) -> list[dict]:
    """Read labeled CSV and return qrels list.

    Each entry:
        {
            "qid": "q23",
            "question": "Among all samples...",
            "relevant": ["2311.16502v3:0", "2311.16502v3:3", ...]
        }
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install pandas")

    df = pd.read_csv(csv_path)

    required_cols = {"query_idx", "query_text", "doc_name", "passage_id", "relevance_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    qrels = []
    for query_idx in sorted(df["query_idx"].unique()):
        rows = df[df["query_idx"] == query_idx]
        question = rows.iloc[0]["query_text"]

        relevant = [
            f"{row['doc_name']}:{int(row['passage_id'])}"
            for _, row in rows.iterrows()
            if row["relevance_label"] >= threshold
        ]

        qrels.append({
            "qid": f"q{int(query_idx)}",
            "question": question,
            "relevant": relevant,
        })

    return qrels


def main():
    parser = argparse.ArgumentParser(description="Build qrels from Kaggle labeled CSV")
    parser.add_argument("--csv", type=str, default=None, help="Path to labeled CSV")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output JSON path")
    parser.add_argument("--threshold", type=int, default=RELEVANCE_THRESHOLD,
                        help="Min relevance_label to count as relevant (default: 2)")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else find_latest_csv(KAGGLE_OUTPUTS)
    print(f"Loading: {csv_path}")

    qrels = build_qrels(csv_path, threshold=args.threshold)

    total_relevant = sum(len(q["relevant"]) for q in qrels)
    print(f"Queries: {len(qrels)}")
    print(f"Total relevant passages: {total_relevant}")
    print(f"Avg relevant per query: {total_relevant / len(qrels):.1f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(qrels, indent=2))
    print(f"Saved qrels → {out_path}")


if __name__ == "__main__":
    main()
