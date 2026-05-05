"""Seed Elasticsearch with deterministic chunks for local / CI retrieval eval.

Usage:
    ELASTICSEARCH_URL=http://localhost:9200 uv run python eval/seed_eval_index.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.indexing.elastic_client import ElasticsearchClient  # noqa: E402
from src.indexing.indexer import DocumentIndexer  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_FIXTURE = Path(__file__).parent / "fixtures" / "eval_corpus_chunks.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed rag_documents index for eval")
    parser.add_argument(
        "--fixture",
        default=str(DEFAULT_FIXTURE),
        help="JSON array of chunk dicts (text, chunk_id, filename, document_id, ...)",
    )
    parser.add_argument(
        "--elasticsearch-url",
        default=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
        help="Elasticsearch base URL",
    )
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        logger.error("Fixture not found: %s", fixture_path)
        sys.exit(1)

    raw = json.loads(fixture_path.read_text())
    if not isinstance(raw, list) or not raw:
        logger.error("Fixture must be a non-empty JSON array of chunks")
        sys.exit(1)

    es = ElasticsearchClient(args.elasticsearch_url)
    es.delete_index("rag_documents")
    indexer = DocumentIndexer(es)

    result = indexer.index_chunks(raw)
    logger.info("Seed complete: %s", result)
    if result.get("errors", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
