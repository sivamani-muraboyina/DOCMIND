"""
Text-only chunking for Mode 2 (fixed legal corpus).

Unlike chunker.py (used by Mode 1), this module works directly on raw .txt
files with no PDF page extraction step. Legal contracts in this corpus are
structurally homogeneous (dense prose, occasional numbered clauses), so we
use a single deliberate strategy — recursive chunking — rather than a
per-document classifier like Mode 1's doc_classifier.py.

Each chunk records its character offset range (char_start, char_end) within
the ORIGINAL document text. This is required to evaluate against
LegalBench-RAG's ground truth, which specifies answer locations as exact
character spans, not page numbers.
"""

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

# Reuse the same chunk size / overlap conventions as Mode 1 for consistency
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Separator hierarchy, same recursive-splitting idea as Stage 2.2
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class LegalChunk:
    """A single chunk from the fixed legal corpus, with character-offset metadata."""
    content: str
    file_path: str          # relative path, e.g. "cuad/ABILITYINC_..._SERVICES_AGREEMENT.txt"
    char_start: int          # offset of this chunk's first character in the ORIGINAL file
    char_end: int            # offset of this chunk's last character in the ORIGINAL file
    chunk_id: str = field(default="")  # assigned later, e.g. "cuad_0042"


def _split_recursive(text: str, chunk_size: int, overlap: int, separators: list[str]) -> list[tuple[str, int, int]]:
    """
    Recursively split text on a hierarchy of separators, returning
    (chunk_text, start_offset, end_offset) tuples relative to the input text.

    This mirrors LangChain's RecursiveCharacterTextSplitter behavior but is
    written from scratch so we can track exact character offsets, which
    LangChain's splitter does not expose by default.
    """
    if len(text) <= chunk_size:
        return [(text, 0, len(text))] if text.strip() else []

    separator = separators[0] if separators else ""
    remaining_separators = separators[1:] if len(separators) > 1 else []

    if separator == "":
        # Last resort: hard cut by character count
        pieces = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            pieces.append((text[start:end], start, end))
            start = end - overlap if (end - overlap) > start else end
        return pieces

    parts = text.split(separator)
    pieces = []
    current_chunk = ""
    current_start = 0
    cursor = 0  # tracks position in the ORIGINAL text as we walk through parts

    for i, part in enumerate(parts):
        part_with_sep = part + (separator if i < len(parts) - 1 else "")

        if len(current_chunk) + len(part_with_sep) <= chunk_size:
            if not current_chunk:
                current_start = cursor
            current_chunk += part_with_sep
        else:
            if current_chunk:
                if len(current_chunk) > chunk_size:
                    # This piece is still too big — recurse with the next separator
                    sub_pieces = _split_recursive(current_chunk, chunk_size, overlap, remaining_separators)
                    for sub_text, sub_s, sub_e in sub_pieces:
                        pieces.append((sub_text, current_start + sub_s, current_start + sub_e))
                else:
                    pieces.append((current_chunk, current_start, current_start + len(current_chunk)))

            current_chunk = part_with_sep
            current_start = cursor

        cursor += len(part_with_sep)

    if current_chunk.strip():
        if len(current_chunk) > chunk_size:
            sub_pieces = _split_recursive(current_chunk, chunk_size, overlap, remaining_separators)
            for sub_text, sub_s, sub_e in sub_pieces:
                pieces.append((sub_text, current_start + sub_s, current_start + sub_e))
        else:
            pieces.append((current_chunk, current_start, current_start + len(current_chunk)))

    return pieces


def chunk_legal_document(text: str, file_path: str) -> list[LegalChunk]:
    """
    Chunk a single legal document's raw text into LegalChunk objects,
    each carrying its exact character offset range in the original file.
    """
    raw_pieces = _split_recursive(text, CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS)

    chunks = []
    for content, start, end in raw_pieces:
        cleaned = content.strip()
        if len(cleaned) < 20:  # skip near-empty fragments
            continue
        # Recompute start/end against the CLEANED text's actual position
        # (stripping whitespace can shift offsets slightly; find() recovers exact position)
        real_start = text.find(cleaned, max(0, start - 5))
        real_end = real_start + len(cleaned)
        chunks.append(
            LegalChunk(
                content=cleaned,
                file_path=file_path,
                char_start=real_start,
                char_end=real_end,
            )
        )

    return chunks


def load_and_chunk_corpus(corpus_dir: Path) -> list[LegalChunk]:
    """
    Walk the entire legal corpus folder, chunk every .txt file, and return
    one flat list of LegalChunk objects with chunk_id assigned sequentially.
    """
    txt_files = sorted(corpus_dir.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} legal documents in {corpus_dir}")

    all_chunks: list[LegalChunk] = []
    for file in txt_files:
        text = file.read_text(encoding="utf-8", errors="ignore")
        relative_path = f"cuad/{file.name}"  # matches the JSON's file_path format exactly
        doc_chunks = chunk_legal_document(text, relative_path)
        all_chunks.extend(doc_chunks)

    for idx, chunk in enumerate(all_chunks):
        chunk.chunk_id = f"cuad_{idx:05d}"

    logger.info(f"Produced {len(all_chunks)} chunks from {len(txt_files)} documents")
    return all_chunks