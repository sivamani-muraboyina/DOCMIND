from __future__ import annotations
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from loguru import logger
from sentence_transformers import CrossEncoder

from backend.services.bm25_store import search_bm25
from backend.services.vector_store import search_faiss
from backend.utils.chunker import Chunk

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def load_reranker() -> CrossEncoder:
    logger.info(f"Loading cross-encoder: {RERANKER_MODEL}")
    return CrossEncoder(RERANKER_MODEL, max_length=512)


def retrieve(
    session_id:   str,
    query:        str,
    top_k_dense:  int   = 10,
    top_k_sparse: int   = 10,
    top_k_final:  int   = 5,
    alpha:        float = 0.6,
) -> List[dict]:
    """
    Hybrid retrieval pipeline:
      1. Dense search  (FAISS cosine similarity)
      2. Sparse search (BM25 keyword match)
      3. Score fusion  (Reciprocal Rank Fusion)
      4. Reranking     (cross-encoder)
    """
    dense_hits, sparse_hits = [], []

    try:
        dense_hits = search_faiss(session_id, query, top_k=top_k_dense)
    except KeyError as e:
        logger.error(e)

    try:
        sparse_hits = search_bm25(session_id, query, top_k=top_k_sparse)
    except KeyError as e:
        logger.error(e)

    if not dense_hits and not sparse_hits:
        return []

    fused    = _reciprocal_rank_fusion(dense_hits, sparse_hits, alpha=alpha)
    reranked = _rerank(query, fused) if len(fused) > 1 else fused

    return [
        {
            "content":          chunk.content,
            "source":           chunk.metadata.get("source", "unknown"),
            "page":             chunk.metadata.get("page"),
            "score":            round(score, 4),
            "retrieval_method": method,
        }
        for chunk, score, method in reranked[:top_k_final]
    ]


def _reciprocal_rank_fusion(
    dense:  List[Tuple[Chunk, float]],
    sparse: List[Tuple[Chunk, float]],
    alpha:  float = 0.6,
    k:      int   = 60,
) -> List[Tuple[Chunk, float, str]]:
    dense_rank  = {c.content: r + 1 for r, (c, _) in enumerate(dense)}
    sparse_rank = {c.content: r + 1 for r, (c, _) in enumerate(sparse)}
    chunk_map   = {c.content: c for c, _ in dense + sparse}

    scored = []
    for content in set(dense_rank) | set(sparse_rank):
        dr = dense_rank.get(content)
        sr = sparse_rank.get(content)
        score = (
            alpha       * (1 / (k + dr)) if dr else 0.0
        ) + (
            (1 - alpha) * (1 / (k + sr)) if sr else 0.0
        )
        method = "hybrid" if (dr and sr) else ("dense" if dr else "sparse")
        scored.append((chunk_map[content], score, method))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _rerank(
    query:      str,
    candidates: List[Tuple[Chunk, float, str]],
) -> List[Tuple[Chunk, float, str]]:
    reranker = load_reranker()
    pairs    = [(query, chunk.content) for chunk, _, _ in candidates]
    scores   = reranker.predict(pairs, show_progress_bar=False)

    reranked = [
        (chunk, float(score), method)
        for (chunk, _, method), score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked

# for mode -2
from backend.services.legal_corpus import search_legal_faiss, search_legal_bm25
from backend.services.legal_corpus import search_legal_faiss, search_legal_bm25, find_matching_document


def retrieve_legal(
    query:                    str,
    top_k_dense:              int   = 10,
    top_k_sparse:             int   = 10,
    top_k_final:              int   = 5,
    alpha:                    float = 0.6,
    use_hnsw:                 bool  = False,
    filter_to_query_document: bool  = False,
) -> List[dict]:
    """
    Hybrid retrieval for Mode 2 (fixed legal corpus). Mirrors retrieve()'s
    pipeline exactly — dense + sparse + RRF + rerank — but reads from
    legal_corpus.py instead of vector_store.py/bm25_store.py, and carries
    char_start/char_end instead of page numbers.
    """
    if filter_to_query_document:
        matched_files = find_matching_document(query, top_n=1)
        if matched_files:
            dense_hits  = search_legal_faiss(query, top_k=top_k_dense, use_hnsw=use_hnsw, restrict_to_files=matched_files)
            sparse_hits = search_legal_bm25(query, top_k=top_k_sparse, restrict_to_files=matched_files)
        else:
            dense_hits  = search_legal_faiss(query, top_k=top_k_dense, use_hnsw=use_hnsw)
            sparse_hits = search_legal_bm25(query, top_k=top_k_sparse)
    else:
        dense_hits  = search_legal_faiss(query, top_k=top_k_dense, use_hnsw=use_hnsw)
        sparse_hits = search_legal_bm25(query, top_k=top_k_sparse)

    if not dense_hits and not sparse_hits:
        return []

    fused = _reciprocal_rank_fusion_legal(dense_hits, sparse_hits, alpha=alpha)
    reranked = _rerank_legal(query, fused) if len(fused) > 1 else fused

    return [
        {
            "content":          chunk.content,
            "source":           chunk.file_path,
            "char_start":       chunk.char_start,
            "char_end":         chunk.char_end,
            "score":            round(score, 4),
            "retrieval_method": method,
        }
        for chunk, score, method in reranked[:top_k_final]
    ]


def _reciprocal_rank_fusion_legal(dense, sparse, alpha: float = 0.6, k: int = 60):
    # Use chunk_id as the dedup key instead of content (LegalChunk has chunk_id; Chunk doesn't)
    dense_rank  = {c.chunk_id: r + 1 for r, (c, _) in enumerate(dense)}
    sparse_rank = {c.chunk_id: r + 1 for r, (c, _) in enumerate(sparse)}
    chunk_map   = {c.chunk_id: c for c, _ in dense + sparse}

    scored = []
    for cid in set(dense_rank) | set(sparse_rank):
        dr = dense_rank.get(cid)
        sr = sparse_rank.get(cid)
        score = (
            alpha       * (1 / (k + dr)) if dr else 0.0
        ) + (
            (1 - alpha) * (1 / (k + sr)) if sr else 0.0
        )
        method = "hybrid" if (dr and sr) else ("dense" if dr else "sparse")
        scored.append((chunk_map[cid], score, method))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _rerank_legal(query: str, candidates):
    reranker = load_reranker()  # reuses the SAME cross-encoder, already loaded via lru_cache
    pairs    = [(query, chunk.content) for chunk, _, _ in candidates]
    scores   = reranker.predict(pairs, show_progress_bar=False)

    reranked = [
        (chunk, float(score), method)
        for (chunk, _, method), score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked