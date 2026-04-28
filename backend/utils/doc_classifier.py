from __future__ import annotations
from pathlib import Path
from typing import Tuple

import pdfplumber
from loguru import logger

from backend.models.schemas import DocumentType

TABLE_RATIO_THRESHOLD = 0.15
SHORT_LINE_THRESHOLD  = 0.40
AVG_LINE_LEN_CUTOFF   = 60


def classify(pdf_path: str | Path) -> Tuple[DocumentType, dict]:
    stats = {
        "total_pages": 0,
        "pages_with_tables": 0,
        "table_ratio": 0.0,
        "short_line_ratio": 0.0,
        "avg_line_length": 0.0,
    }
    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_lines, table_pages = [], 0
            for page in pdf.pages:
                tables = page.extract_tables() or []
                if any(len(t) > 1 for t in tables):
                    table_pages += 1
                text  = page.extract_text() or ""
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                all_lines.extend(lines)

            total         = len(pdf.pages)
            table_ratio   = table_pages / max(total, 1)
            short_ratio   = len([l for l in all_lines if len(l) < 50]) / max(len(all_lines), 1)
            avg_len       = sum(len(l) for l in all_lines) / max(len(all_lines), 1)

            stats.update({
                "total_pages": total,
                "pages_with_tables": table_pages,
                "table_ratio": round(table_ratio, 3),
                "short_line_ratio": round(short_ratio, 3),
                "avg_line_length": round(avg_len, 1),
            })

            if table_ratio >= TABLE_RATIO_THRESHOLD or short_ratio >= SHORT_LINE_THRESHOLD:
                doc_type = DocumentType.STRUCTURED
            elif short_ratio < 0.20 and avg_len > 80:
                doc_type = DocumentType.UNSTRUCTURED
            else:
                doc_type = DocumentType.MIXED

    except Exception as exc:
        logger.warning(f"Classification error ({exc}), defaulting to MIXED")
        doc_type = DocumentType.MIXED

    logger.info(f"Classified as {doc_type.value} | {stats}")
    return doc_type, stats
