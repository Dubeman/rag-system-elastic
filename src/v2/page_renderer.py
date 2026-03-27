"""Render PDF bytes to page images (vision-native ingestion)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PageImageRecord:
    """One rendered page with stable identifiers for retrieval and citations."""

    doc_id: str
    page_num: int
    image_path: str
    width: int
    height: int
    sha256: str


def _doc_id_from_name(filename: str, content: bytes) -> str:
    h = hashlib.sha256(content).hexdigest()[:16]
    safe = "".join(c if c.isalnum() else "_" for c in (filename or "doc"))[:40]
    return f"{safe}_{h}"


def render_pdf_to_pages(
    pdf_content: bytes,
    *,
    filename: str,
    out_dir: Path,
    dpi: int = 144,
    image_format: str = "png",
) -> List[PageImageRecord]:
    """
    Rasterize each PDF page to an image file under out_dir/doc_id/.

    Requires PyMuPDF (``import fitz``).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError("PyMuPDF (pymupdf) is required for v2 page rendering") from e

    doc_id = _doc_id_from_name(filename, pdf_content)
    doc_dir = out_dir / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(stream=pdf_content, filetype="pdf")
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    records: List[PageImageRecord] = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        fname = f"page_{page_num:04d}.{image_format}"
        path = doc_dir / fname
        pix.save(str(path))

        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        records.append(
            PageImageRecord(
                doc_id=doc_id,
                page_num=page_num,
                image_path=str(path.resolve()),
                width=pix.width,
                height=pix.height,
                sha256=sha,
            )
        )

    doc.close()
    logger.info("Rendered %s pages for doc_id=%s under %s", len(records), doc_id, doc_dir)
    return records


def create_minimal_pdf_bytes(text: str) -> bytes:
    """Build a one-page PDF in memory (for sample v2 ingest)."""
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError("PyMuPDF (pymupdf) is required for v2 sample PDF creation") from e

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text[:8000] or "empty")
    out = doc.tobytes()
    doc.close()
    return out
