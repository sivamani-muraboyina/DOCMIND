from __future__ import annotations
import os, uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from loguru import logger

from backend.models.schemas import DocumentType, UploadResponse
from backend.services.bm25_store import build_bm25_index
from backend.services.pdf_processor import extract_pages
from backend.services.vector_store import build_faiss_index
from backend.utils.chunker import chunk_document
from backend.utils.doc_classifier import classify

router     = APIRouter(prefix="/upload", tags=["Upload"])
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
MAX_MB     = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

CHUNK_SIZES = {
    DocumentType.STRUCTURED:   int(os.getenv("CHUNK_SIZE_STRUCTURED", "256")),
    DocumentType.UNSTRUCTURED: int(os.getenv("CHUNK_SIZE_DEFAULT",    "512")),
    DocumentType.MIXED:        int(os.getenv("CHUNK_SIZE_DEFAULT",    "512")),
}
CHUNK_OVERLAPS = {
    DocumentType.STRUCTURED:   int(os.getenv("CHUNK_OVERLAP_STRUCTURED", "50")),
    DocumentType.UNSTRUCTURED: int(os.getenv("CHUNK_OVERLAP_DEFAULT",    "100")),
    DocumentType.MIXED:        int(os.getenv("CHUNK_OVERLAP_DEFAULT",    "100")),
}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only PDF files are accepted.")

    session_id = str(uuid.uuid4())
    save_path  = UPLOAD_DIR / f"{session_id}_{Path(file.filename).name}"

    raw = await file.read()
    if len(raw) > MAX_MB * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {MAX_MB} MB.")

    async with aiofiles.open(save_path, "wb") as f:
        await f.write(raw)
    logger.info(f"Saved {save_path.name} ({len(raw) / 1024:.1f} KB)")

    try:
        doc_type, _ = classify(save_path)
    except Exception:
        doc_type = DocumentType.MIXED

    chunk_size    = CHUNK_SIZES[doc_type]
    chunk_overlap = CHUNK_OVERLAPS[doc_type]

    pages, num_pages = extract_pages(save_path, doc_type)
    if not any(p["text"].strip() for p in pages):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "No extractable text found in PDF.")

    chunks, strategy = chunk_document(pages, doc_type, chunk_size, chunk_overlap)
    if not chunks:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Document produced zero chunks.")

    n_indexed = build_faiss_index(session_id, chunks)
    try:
        build_bm25_index(session_id, chunks)
    except Exception as exc:
        logger.warning(f"BM25 index failed (non-fatal): {exc}")

    logger.info(f"Indexed | session={session_id} | type={doc_type.value} | chunks={n_indexed}")

    return UploadResponse(
        session_id     = session_id,
        filename       = file.filename,
        num_pages      = num_pages,
        num_chunks     = n_indexed,
        doc_type       = doc_type,
        chunk_strategy = strategy,
        chunk_size     = chunk_size,
        chunk_overlap  = chunk_overlap,
    )
