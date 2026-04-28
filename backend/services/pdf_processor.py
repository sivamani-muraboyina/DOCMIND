from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import fitz
import pdfplumber
from loguru import logger

from backend.models.schemas import DocumentType


def extract_pages(pdf_path: str | Path, doc_type: DocumentType) -> Tuple[List[dict], int]:
    path   = Path(pdf_path)
    source = path.name

    extractor = _plumber_extract if doc_type == DocumentType.STRUCTURED else _mupdf_extract
    pages = extractor(path, source)

    if not any(p["text"].strip() for p in pages):
        logger.warning(f"Primary extractor empty for {source}, trying fallback")
        fallback = _mupdf_extract if doc_type == DocumentType.STRUCTURED else _plumber_extract
        pages = fallback(path, source)

    logger.info(f"Extracted {len(pages)} pages from {source}")
    return pages, len(pages)


def _plumber_extract(path: Path, source: str) -> List[dict]:
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for n, page in enumerate(pdf.pages, 1):
                parts = []
                for table in (page.extract_tables() or []):
                    row_text = _table_to_text(table)
                    if row_text:
                        parts.append(row_text)
                raw = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                if raw.strip():
                    parts.append(raw)
                pages.append({"text": "\n\n".join(parts), "page": n, "source": source})
    except Exception as exc:
        logger.error(f"pdfplumber failed: {exc}")
    return pages


def _mupdf_extract(path: Path, source: str) -> List[dict]:
    pages = []
    try:
        doc = fitz.open(str(path))
        for n, page in enumerate(doc, 1):
            blocks = page.get_text("blocks", sort=True)
            text   = "\n\n".join(b[4].strip() for b in blocks if b[6] == 0 and b[4].strip())
            pages.append({"text": text, "page": n, "source": source})
        doc.close()
    except Exception as exc:
        logger.error(f"PyMuPDF failed: {exc}")
    return pages


def _table_to_text(table: list) -> str:
    rows = [" | ".join(str(cell or "").strip() for cell in row) for row in table]
    return "\n".join(rows)
