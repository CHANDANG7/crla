"""
ZECRO-RL — PDF Extractor for TA Books
Extracts text and page metadata from PDF books.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class PDFExtractor:
    """Extracts text from PDF books page-by-page."""

    def extract(self, file_path: str | Path) -> List[Dict[str, Any]]:
        """
        Extract pages from PDF.
        Returns list of dicts: [{"page": 1, "text": "...", "book": "title.pdf"}]
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("PDF file not found", path=str(path))
            return []

        pages = []
        book_title = path.stem.replace("_", " ").title()

        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                cleaned = self._clean_text(text)
                if len(cleaned) > 50:  # Skip blank or trivial pages
                    pages.append({
                        "book": book_title,
                        "file_name": path.name,
                        "page": page_num + 1,
                        "text": cleaned,
                    })
            doc.close()
            logger.info("Extracted PDF pages", book=book_title, pages=len(pages))
        except ImportError:
            logger.warning("PyMuPDF (fitz) not installed. Trying fallback text reader.")
            try:
                import pypdf
                reader = pypdf.PdfReader(str(path))
                for idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    cleaned = self._clean_text(text)
                    if len(cleaned) > 50:
                        pages.append({
                            "book": book_title,
                            "file_name": path.name,
                            "page": idx + 1,
                            "text": cleaned,
                        })
            except Exception as e:
                logger.error("Fallback PDF extraction failed", error=str(e))

        return pages

    def _clean_text(self, text: str) -> str:
        """Remove excess whitespace, page numbering artifacts."""
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


pdf_extractor = PDFExtractor()
