from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List

import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from backend.models.schemas import ChunkStrategy, DocumentType

for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


@dataclass
class Chunk:
    content:  str
    metadata: dict


_STRATEGY_MAP = {
    DocumentType.STRUCTURED:   ChunkStrategy.FIXED_SIZE,
    DocumentType.UNSTRUCTURED: ChunkStrategy.RECURSIVE,
    DocumentType.MIXED:        ChunkStrategy.SEMANTIC,
}


def chunk_document(
    pages:         List[dict],
    doc_type:      DocumentType,
    chunk_size:    int = 512,
    chunk_overlap: int = 100,
) -> tuple[List[Chunk], ChunkStrategy]:
    strategy = _STRATEGY_MAP[doc_type]
    logger.info(f"Chunking | strategy={strategy.value} | pages={len(pages)}")

    dispatch = {
        ChunkStrategy.FIXED_SIZE: _fixed_chunks,
        ChunkStrategy.RECURSIVE:  _recursive_chunks,
        ChunkStrategy.SEMANTIC:   _semantic_chunks,
    }
    return dispatch[strategy](pages, chunk_size, chunk_overlap), strategy


def _make_splitter(chunk_size: int, overlap: int, separators: List[str]) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators,
        length_function=len,
    )


def _apply_splitter(splitter, pages: List[dict], strategy: ChunkStrategy) -> List[Chunk]:
    chunks, idx = [], 0
    for p in pages:
        text = p["text"].strip()
        if not text:
            continue
        for split in splitter.split_text(text):
            split = split.strip()
            if split:
                chunks.append(Chunk(
                    content=split,
                    metadata={"source": p.get("source", "doc"), "page": p.get("page"), "chunk_index": idx, "strategy": strategy.value}
                ))
                idx += 1
    return chunks


def _fixed_chunks(pages: List[dict], size: int, overlap: int) -> List[Chunk]:
    return _apply_splitter(
        _make_splitter(size, overlap, ["\n\n", "\n", " | ", "|", " ", ""]),
        pages, ChunkStrategy.FIXED_SIZE
    )


def _recursive_chunks(pages: List[dict], size: int, overlap: int) -> List[Chunk]:
    return _apply_splitter(
        _make_splitter(size, overlap, ["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""]),
        pages, ChunkStrategy.RECURSIVE
    )


def _semantic_chunks(pages: List[dict], size: int, overlap: int) -> List[Chunk]:
    chunks, overlap_n = [], max(1, overlap // 80)

    for p in pages:
        text = p["text"].strip()
        if not text:
            continue
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = re.split(r"(?<=[.!?])\s+", text)

        current, cur_len, idx = [], 0, 0
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if cur_len + len(sent) > size and current:
                chunks.append(Chunk(
                    content=" ".join(current),
                    metadata={"source": p.get("source", "doc"), "page": p.get("page"), "chunk_index": idx, "strategy": ChunkStrategy.SEMANTIC.value}
                ))
                idx   += 1
                current = current[-overlap_n:]
                cur_len = sum(len(s) for s in current)
            current.append(sent)
            cur_len += len(sent)

        if current:
            chunks.append(Chunk(
                content=" ".join(current),
                metadata={"source": p.get("source", "doc"), "page": p.get("page"), "chunk_index": idx, "strategy": ChunkStrategy.SEMANTIC.value}
            ))

    logger.info(f"Semantic chunker → {len(chunks)} chunks")
    return chunks
