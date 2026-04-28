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
