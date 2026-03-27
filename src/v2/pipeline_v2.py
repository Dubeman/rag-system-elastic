"""Orchestrate v2 ingest (page images) and query (vision retrieval + VLM)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .page_renderer import create_minimal_pdf_bytes, render_pdf_to_pages
from .vlm_client import RunPodVLMClient
from .vision_retrieval import VisionRetriever

logger = logging.getLogger(__name__)


class VisionPipelineV2:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.pages_root = self.data_dir / "pages"
        self.pages_root.mkdir(parents=True, exist_ok=True)
        self.retriever = VisionRetriever(self.data_dir)
        self.vlm = RunPodVLMClient()

    def ingest_pdf_bytes(self, content: bytes, filename: str) -> Dict[str, Any]:
        records = render_pdf_to_pages(
            content, filename=filename, out_dir=self.pages_root
        )
        meta = [
            {
                "doc_id": r.doc_id,
                "page_num": r.page_num,
                "image_path": r.image_path,
                "filename": filename,
            }
            for r in records
        ]
        n = self.retriever.index_page_records(meta)
        return {
            "doc_id": records[0].doc_id if records else "",
            "pages_rendered": len(records),
            "vectors_indexed": n,
            "records": meta,
        }

    def ingest_sample_text(self, text: str, filename: str = "sample_v2.pdf") -> Dict[str, Any]:
        pdf_bytes = create_minimal_pdf_bytes(text)
        return self.ingest_pdf_bytes(pdf_bytes, filename)

    def ingest_from_google_drive(
        self,
        folder_id: str,
        ingestion: Any,
    ) -> Dict[str, Any]:
        """Fetch PDFs using the same Drive client as v1, then render and index pages."""
        if not ingestion.google_client:
            ingestion.initialize_google_client()
        if not ingestion.google_client:
            raise ValueError("Google Drive client unavailable")

        if hasattr(ingestion.google_client, "fetch_pdfs_from_public_folder"):
            pdf_files = ingestion.google_client.fetch_pdfs_from_public_folder(folder_id)
        else:
            pdf_files = ingestion.google_client.fetch_pdfs_from_folder(folder_id)

        totals = {"documents": 0, "pages_rendered": 0, "vectors_indexed": 0}
        details: List[Dict[str, Any]] = []
        for pdf in pdf_files:
            name = pdf.get("name", "unknown.pdf")
            content = pdf.get("content", b"")
            if not content:
                continue
            r = self.ingest_pdf_bytes(content, name)
            totals["documents"] += 1
            totals["pages_rendered"] += r["pages_rendered"]
            totals["vectors_indexed"] += r["vectors_indexed"]
            details.append(r)
        return {"status": "success", "totals": totals, "details": details}

    def query(
        self,
        question: str,
        top_k: int = 5,
        generate_answer: bool = True,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        results = self.retriever.search(question, top_k=top_k)
        retrieve_ms = (time.perf_counter() - t0) * 1000.0

        llm_response: Optional[Dict[str, Any]] = None
        gen_ms = 0.0
        if generate_answer:
            t1 = time.perf_counter()
            llm_response = self.vlm.generate_with_pages(question, results)
            gen_ms = (time.perf_counter() - t1) * 1000.0

        return {
            "question": question,
            "search_mode": "vision_colpali_faiss",
            "results": results,
            "total_results": len(results),
            "status": "success",
            "pipeline_version": "v2",
            "llm_response": llm_response,
            "timings_ms": {
                "retrieve": round(retrieve_ms, 3),
                "generate": round(gen_ms, 3),
            },
        }
