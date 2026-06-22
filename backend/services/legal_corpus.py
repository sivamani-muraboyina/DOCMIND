"""
Mode 2 — fixed legal corpus (CUAD), built once and cached to disk.

Mirrors vector_store.py's FAISS conventions, but for a single persistent
index instead of per-session ones. Supports both IndexFlatIP and
IndexHNSWFlat so the same corpus can be benchmarked under both (Stage 1.3 /
Stage 4.9's latency optimization work).

Build artifacts are written to data/legal_index/ — gitignored, never
committed. First run builds everything from data/raw/corpus/cuad/; later
runs load the cached files directly.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from backend.services.embeddings import embed_texts, embed_query
from backend.utils.legal_chunker import LegalChunk, load_and_chunk_corpus

CORPUS_DIR = Path("data/raw/corpus/cuad")
INDEX_DIR = Path("data/legal_index")

FAISS_FLAT_PATH = INDEX_DIR / "legal_faiss_flat.index"
FAISS_HNSW_PATH = INDEX_DIR / "legal_faiss_hnsw.index"
BM25_PATH = INDEX_DIR / "legal_bm25.pkl"
CHUNKS_PATH = INDEX_DIR / "legal_chunks.pkl"

# In-memory cache once loaded at startup — avoids re-reading disk every request
_legal_store: dict = {}


def _build_everything() -> Tuple[faiss.Index, faiss.Index, BM25Okapi, List[LegalChunk]]:
    """Run the full indexing phase: chunk corpus, embed, build both FAISS variants + BM25."""
    logger.info("Building legal corpus index from scratch (first run)...")

    chunks = load_and_chunk_corpus(CORPUS_DIR)
    texts = [c.content for c in chunks]

    embeddings = embed_texts(texts)
    dim = embeddings.shape[1]

    # Flat — exact search, baseline for the benchmark
    flat_index = faiss.IndexFlatIP(dim)
    flat_index.add(embeddings)

    # HNSW — approximate search, the comparison point
    hnsw_index = faiss.IndexHNSWFlat(dim, 32)  # M=32, Stage 1.3 convention
    hnsw_index.hnsw.efConstruction = 40
    hnsw_index.add(embeddings)

    # BM25 — sparse retrieval, same tokenization style as Mode 1's bm25_store.py
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    return flat_index, hnsw_index, bm25, chunks


def _save_to_disk(flat_index, hnsw_index, bm25, chunks: List[LegalChunk]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(flat_index, str(FAISS_FLAT_PATH))
    faiss.write_index(hnsw_index, str(FAISS_HNSW_PATH))

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    logger.info(f"Saved legal index artifacts to {INDEX_DIR}")


def _load_from_disk() -> Tuple[faiss.Index, faiss.Index, BM25Okapi, List[LegalChunk]]:
    logger.info(f"Loading cached legal index from {INDEX_DIR}...")

    flat_index = faiss.read_index(str(FAISS_FLAT_PATH))
    hnsw_index = faiss.read_index(str(FAISS_HNSW_PATH))

    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return flat_index, hnsw_index, bm25, chunks


def initialize_legal_corpus(force_rebuild: bool = False) -> None:
    """
    Call this once at FastAPI startup (in main.py's lifespan).
    Builds from raw documents only if no cached index exists yet, or if
    force_rebuild=True (e.g. after changing chunk size for a new benchmark run).
    """
    already_cached = all(p.exists() for p in [FAISS_FLAT_PATH, FAISS_HNSW_PATH, BM25_PATH, CHUNKS_PATH])

    if already_cached and not force_rebuild:
        flat_index, hnsw_index, bm25, chunks = _load_from_disk()
    else:
        flat_index, hnsw_index, bm25, chunks = _build_everything()
        _save_to_disk(flat_index, hnsw_index, bm25, chunks)

    _legal_store["flat_index"] = flat_index
    _legal_store["hnsw_index"] = hnsw_index
    _legal_store["bm25"] = bm25
    _legal_store["chunks"] = chunks

    logger.info(
        f"Legal corpus ready | chunks={len(chunks)} | "
        f"flat_vectors={flat_index.ntotal} | hnsw_vectors={hnsw_index.ntotal}"
    )


def search_legal_faiss(query: str, top_k: int = 10, use_hnsw: bool = False, restrict_to_files: list[str] | None = None) -> List[Tuple[LegalChunk, float]]:
    if "chunks" not in _legal_store:
        raise RuntimeError("Legal corpus not initialized — call initialize_legal_corpus() first")

    index = _legal_store["hnsw_index"] if use_hnsw else _legal_store["flat_index"]
    chunks = _legal_store["chunks"]

    q_vec = embed_query(query)

    if restrict_to_files:
        # Search broadly first, then filter — but pull a MUCH larger pool so
        # restricted documents actually have a chance to surface
        search_k = min(500, index.ntotal)
        scores, idxs = index.search(q_vec, search_k)
        results = [
            (chunks[i], float(s))
            for s, i in zip(scores[0], idxs[0])
            if i != -1 and chunks[i].file_path in restrict_to_files
        ]
        return results[:top_k]

    scores, idxs = index.search(q_vec, min(top_k, index.ntotal))
    return [
        (chunks[i], float(s))
        for s, i in zip(scores[0], idxs[0])
        if i != -1
    ]



def search_legal_bm25(query: str, top_k: int = 10, restrict_to_files: list[str] | None = None) -> List[Tuple[LegalChunk, float]]:
    if "chunks" not in _legal_store:
        raise RuntimeError("Legal corpus not initialized — call initialize_legal_corpus() first")

    bm25 = _legal_store["bm25"]
    chunks = _legal_store["chunks"]

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    if restrict_to_files:
        candidate_indices = [i for i, c in enumerate(chunks) if c.file_path in restrict_to_files]
        candidate_indices.sort(key=lambda i: scores[i], reverse=True)
        return [(chunks[i], float(scores[i])) for i in candidate_indices[:top_k]]

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_indices]


def get_legal_corpus_stats() -> dict:
    """Quick stats for sanity-checking and for your README/benchmark reporting."""
    if "chunks" not in _legal_store:
        return {"initialized": False}
    return {
        "initialized": True,
        "total_chunks": len(_legal_store["chunks"]),
        "flat_vectors": _legal_store["flat_index"].ntotal,
        "hnsw_vectors": _legal_store["hnsw_index"].ntotal,
    }

import re


from rank_bm25 import BM25Okapi

_filename_bm25_cache = {}


def find_matching_document(query: str, top_n: int = 1, min_confidence: int = 2) -> list[str]:
    """
    Extract capitalized multi-word phrases (likely company/entity names) from
    the question, find filenames containing those fragments, and only return
    a match if it's backed by at least `min_confidence` distinct name matches
    — a single matched word (e.g. one company name) is too easy to mismatch
    against an unrelated filename that happens to share that one term.
    """
    if "chunks" not in _legal_store:
        raise RuntimeError("Legal corpus not initialized")

    all_files = sorted({c.file_path for c in _legal_store["chunks"]})

    candidate_names = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", query)
    candidate_names = [n for n in candidate_names if len(n) > 4]

    scored = []
    for f in all_files:
        f_clean = f.lower().replace(" ", "").replace("_", "").replace(",", "").replace(".", "")
        match_count = sum(
            1 for name in candidate_names
            if name.lower().replace(" ", "") in f_clean
        )
        if match_count > 0:
            scored.append((f, match_count))

    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored or scored[0][1] < min_confidence:
        return []  # not confident enough — caller should fall back to unfiltered

    return [f for f, _ in scored[:top_n]]