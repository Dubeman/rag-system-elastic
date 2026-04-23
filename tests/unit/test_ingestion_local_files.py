from pathlib import Path
from unittest.mock import MagicMock

from ingestion.pipeline import IngestionPipeline


def _touch(path: Path, content: bytes = b"fake pdf bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_expand_local_paths_with_explicit_pdf(tmp_path):
    pipeline = IngestionPipeline()
    pdf_path = _touch(tmp_path / "doc1.pdf")

    expanded = pipeline._expand_local_pdf_paths([str(pdf_path)])

    assert [p.resolve() for p in expanded] == [pdf_path.resolve()]


def test_expand_local_paths_with_directory(tmp_path):
    pipeline = IngestionPipeline()
    docs_dir = tmp_path / "docs"
    pdf_a = _touch(docs_dir / "a.pdf")
    pdf_b = _touch(docs_dir / "b.pdf")
    _touch(docs_dir / "notes.txt", b"not a pdf")

    expanded = pipeline._expand_local_pdf_paths([str(docs_dir)])
    resolved = {p.resolve() for p in expanded}

    assert resolved == {pdf_a.resolve(), pdf_b.resolve()}


def test_expand_local_paths_with_glob(tmp_path):
    pipeline = IngestionPipeline()
    docs_dir = tmp_path / "docs"
    pdf_a = _touch(docs_dir / "a.pdf")
    pdf_b = _touch(docs_dir / "b.pdf")

    expanded = pipeline._expand_local_pdf_paths([str(docs_dir / "*.pdf")])
    resolved = {p.resolve() for p in expanded}

    assert resolved == {pdf_a.resolve(), pdf_b.resolve()}


def test_expand_local_paths_mixed_inputs_are_deduplicated(tmp_path):
    pipeline = IngestionPipeline()
    docs_dir = tmp_path / "docs"
    pdf_a = _touch(docs_dir / "a.pdf")
    _touch(docs_dir / "b.pdf")

    expanded = pipeline._expand_local_pdf_paths(
        [str(pdf_a), str(docs_dir), str(docs_dir / "*.pdf")]
    )
    resolved = {p.resolve() for p in expanded}

    assert len(expanded) == 2
    assert resolved == {
        (docs_dir / "a.pdf").resolve(),
        (docs_dir / "b.pdf").resolve(),
    }


def test_ingest_from_local_files_returns_empty_when_no_valid_matches(tmp_path):
    pipeline = IngestionPipeline()
    pipeline.pdf_parser.parse_multiple_pdfs = MagicMock(return_value=[])

    result = pipeline.ingest_from_local_files([str(tmp_path / "*.pdf")])

    assert result == []
    pipeline.pdf_parser.parse_multiple_pdfs.assert_not_called()
