"""PDF text extraction utilities."""

import io
import logging
from typing import Dict, List, Optional

import pypdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

logger = logging.getLogger(__name__)


class PDFParser:
    """PDF text extraction with multiple parsing strategies."""

    def __init__(self, fallback_enabled: bool = True):
        """
        Initialize PDF parser.

        Args:
            fallback_enabled: Whether to use fallback parsing methods
        """
        self.fallback_enabled = fallback_enabled

    def extract_text_pypdf(self, pdf_content: bytes) -> str:
        """
        Extract text using PyPDF library.

        Args:
            pdf_content: PDF file content as bytes

        Returns:
            Extracted text content
        """
        try:
            pdf_stream = io.BytesIO(pdf_content)
            reader = pypdf.PdfReader(pdf_stream)

            text_content = []
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue

            full_text = "\n\n".join(text_content)
            logger.debug(f"Extracted {len(full_text)} characters using PyPDF")
            return full_text

        except Exception as e:
            logger.error(f"PyPDF extraction failed: {e}")
            if self.fallback_enabled:
                return self.extract_text_pdfminer(pdf_content)
            raise

    def extract_text_pdfminer(self, pdf_content: bytes) -> str:
        """
        Extract text using PDFMiner library (fallback method).

        Args:
            pdf_content: PDF file content as bytes

        Returns:
            Extracted text content
        """
        try:
            pdf_stream = io.BytesIO(pdf_content)
            parser = PDFParser(pdf_stream)
            document = PDFDocument(parser)

            if not document.is_extractable:
                logger.warning("PDF is not extractable")
                return ""

            resource_manager = PDFResourceManager()
            output_stream = io.StringIO()
            converter = TextConverter(
                resource_manager, output_stream, laparams=LAParams()
            )
            interpreter = PDFPageInterpreter(resource_manager, converter)

            for page in PDFPage.create_pages(document):
                interpreter.process_page(page)

            text = output_stream.getvalue()
            converter.close()
            output_stream.close()

            logger.debug(f"Extracted {len(text)} characters using PDFMiner")
            return text

        except Exception as e:
            logger.error(f"PDFMiner extraction failed: {e}")
            raise

    def extract_text(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF using the best available method.

        Args:
            pdf_content: PDF file content as bytes

        Returns:
            Extracted text content
        """
        # Try PyPDF first (faster)
        try:
            text = self.extract_text_pypdf(pdf_content)
            if text.strip():
                return text
        except Exception:
            pass

        # Fallback to PDFMiner if enabled
        if self.fallback_enabled:
            try:
                return self.extract_text_pdfminer(pdf_content)
            except Exception as e:
                logger.error(f"All PDF extraction methods failed: {e}")
                return ""

        return ""

    def extract_metadata(self, pdf_content: bytes) -> Dict:
        """
        Extract metadata from PDF.

        Args:
            pdf_content: PDF file content as bytes

        Returns:
            Dictionary containing PDF metadata
        """
        metadata = {
            "pages": 0,
            "title": "",
            "author": "",
            "subject": "",
            "creator": "",
            "producer": "",
            "creation_date": "",
            "modification_date": "",
        }

        try:
            pdf_stream = io.BytesIO(pdf_content)
            reader = pypdf.PdfReader(pdf_stream)

            metadata["pages"] = len(reader.pages)

            if reader.metadata:
                pdf_metadata = reader.metadata
                metadata.update(
                    {
                        "title": pdf_metadata.get("/Title", ""),
                        "author": pdf_metadata.get("/Author", ""),
                        "subject": pdf_metadata.get("/Subject", ""),
                        "creator": pdf_metadata.get("/Creator", ""),
                        "producer": pdf_metadata.get("/Producer", ""),
                        "creation_date": str(pdf_metadata.get("/CreationDate", "")),
                        "modification_date": str(
                            pdf_metadata.get("/ModDate", "")
                        ),
                    }
                )

        except Exception as e:
            logger.error(f"Failed to extract PDF metadata: {e}")

        return metadata

    def parse_pdf(self, pdf_content: bytes, filename: str = "") -> Dict:
        """
        Parse PDF and extract both text and metadata.

        Args:
            pdf_content: PDF file content as bytes
            filename: Original filename for reference

        Returns:
            Dictionary containing extracted text and metadata
        """
        logger.info(f"Parsing PDF: {filename}")

        try:
            text = self.extract_text(pdf_content)
            metadata = self.extract_metadata(pdf_content)

            result = {
                "filename": filename,
                "text": text,
                "char_count": len(text),
                "metadata": metadata,
                "extraction_success": bool(text.strip()),
            }

            logger.info(
                f"PDF parsing complete: {result['char_count']} characters extracted"
            )
            return result

        except Exception as e:
            logger.error(f"PDF parsing failed for {filename}: {e}")
            return {
                "filename": filename,
                "text": "",
                "char_count": 0,
                "metadata": {},
                "extraction_success": False,
                "error": str(e),
            }

    def parse_multiple_pdfs(self, pdf_files: List[Dict]) -> List[Dict]:
        """
        Parse multiple PDF files.

        Args:
            pdf_files: List of dictionaries containing PDF data

        Returns:
            List of parsed PDF results
        """
        results = []

        for pdf_file in pdf_files:
            try:
                result = self.parse_pdf(
                    pdf_file["content"], pdf_file.get("name", "unknown.pdf")
                )

                # Add original file metadata
                result.update(
                    {
                        "file_id": pdf_file.get("id", ""),
                        "file_size": pdf_file.get("size", 0),
                        "file_url": pdf_file.get("url", ""),
                        "modified_time": pdf_file.get("modified_time", ""),
                    }
                )

                results.append(result)

            except Exception as e:
                logger.error(f"Failed to parse PDF {pdf_file.get('name', 'unknown')}: {e}")
                results.append(
                    {
                        "filename": pdf_file.get("name", "unknown.pdf"),
                        "text": "",
                        "char_count": 0,
                        "metadata": {},
                        "extraction_success": False,
                        "error": str(e),
                    }
                )

        successful_parses = sum(1 for r in results if r["extraction_success"])
        logger.info(f"Successfully parsed {successful_parses}/{len(results)} PDFs")

        return results
