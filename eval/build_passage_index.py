"""Build passage-aligned Elasticsearch index from Kaggle labeled CSV (eval only).

One indexed document per (doc_name, passage_id) with the same ID space as qrels:
  canonical_id = "{doc_name}:{passage_id}"

Uses the same dense + ELSER embedding pipeline as v1 via DocumentIndexer.

Usage:
    python eval/build_passage_index.py
    python eval/build_passage_index.py --csv /path/to.csv --index rag_passages_v2
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

# Repo root on path (for `src.*` imports)
sys.path.insert(0, str(Path(__file__).parent.parent))

def _load_build_qrels():
    bq_path = Path(__file__).parent / "build_qrels.py"
    spec = importlib.util.spec_from_file_location("build_qrels", bq_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_bq = _load_build_qrels()
find_latest_csv = _bq.find_latest_csv
_pick_column = _bq._pick_column
PASSAGE_TEXT_CANDIDATES = _bq.PASSAGE_TEXT_CANDIDATES

from src.indexing.elastic_client import ElasticsearchClient  # noqa: E402
from src.indexing.indexer import DocumentIndexer  # noqa: E402

KAGGLE_OUTPUTS = Path(__file__).parent.parent.parent / "kaggle" / "outputs"
DEFAULT_INDEX = "rag_passages_v2"


def _normalize_doc_stem(doc_name: str) -> str:
    s = str(doc_name).strip()
    if s.lower().endswith(".pdf"):
        return Path(s).stem
    return s


def _filename_for_doc(doc_name: str) -> str:
    s = str(doc_name).strip()
    if s.lower().endswith(".pdf"):
        return Path(s).name
    return f"{s}.pdf"


def load_passage_table(csv_path: Path) -> tuple[list[dict], str]:
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas required: pip install pandas") from e

    df = pd.read_csv(csv_path)
    col = _pick_column(df, PASSAGE_TEXT_CANDIDATES)
    if not col:
        print(
            "[ERROR] No passage text column found "
            f"(tried {PASSAGE_TEXT_CANDIDATES}). passage-aligned index disabled."
        )
        sys.exit(1)

    required = {"doc_name", "passage_id", col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    sub = df[["doc_name", "passage_id", col]].copy()
    sub = sub.dropna(subset=[col])
    sub = sub[sub[col].astype(str).str.strip() != ""]

    # One passage per (doc_name, passage_id); keep last row in file order
    sub = sub.drop_duplicates(subset=["doc_name", "passage_id"], keep="last")

    chunks: list[dict] = []
    for _, row in sub.iterrows():
        text = row[col]
        if text is None or (isinstance(text, float) and str(text) == "nan"):
            continue
        text_s = str(text).strip()
        if not text_s:
            continue
        doc_raw = row["doc_name"]
        pid = int(row["passage_id"])
        stem = _normalize_doc_stem(str(doc_raw))
        canonical_id = f"{stem}:{pid}"
        fname = _filename_for_doc(str(doc_raw))
        chunks.append(
            {
                "canonical_id": canonical_id,
                "document_id": stem,
                "chunk_id": pid,
                "filename": fname,
                "text": text_s,
                "source_url": "",
                "file_url": "",
                "modified_time": "",
                "token_count": max(1, len(text_s) // 4),
                "char_count": len(text_s),
            }
        )

    return chunks, col


def main() -> None:
    parser = argparse.ArgumentParser(description="Index passage-aligned eval corpus")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--index", type=str, default=DEFAULT_INDEX, help="Elasticsearch index name")
    parser.add_argument(
        "--es-url",
        type=str,
        default=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
        help="Elasticsearch base URL",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else find_latest_csv(KAGGLE_OUTPUTS)
    print(f"Loading passages from: {csv_path}")

    chunks, col_used = load_passage_table(csv_path)
    print(f"Passage text column: {col_used}")
    print(f"Passages to index: {len(chunks)}")

    if not chunks:
        print("[ERROR] No non-empty passages to index.")
        sys.exit(1)

    es_client = ElasticsearchClient(args.es_url)
    indexer = DocumentIndexer(es_client, index_name=args.index)
    result = indexer.index_chunks(chunks)
    print(f"Done. indexed={result['indexed']} errors={result['errors']} index={args.index}")


if __name__ == "__main__":
    main()
