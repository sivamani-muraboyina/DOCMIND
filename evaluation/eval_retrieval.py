"""
Retrieval evaluation — Precision@k, Recall@k, MRR.

Runs two separate comparisons:
1. Dense-only: Flat vs HNSW (isolates the index-type tradeoff, Stage 1.3)
2. Full pipeline: hybrid+rerank, using whichever index is configured
   (this is the number that reflects what users actually experience)

Read-only against the existing legal corpus index — does not modify it.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

from loguru import logger

from backend.services.legal_corpus import initialize_legal_corpus, search_legal_faiss
from backend.services.retriever import retrieve_legal

from evaluation.build_test_set import EvalQuestion

TEST_SET_PATH = Path("evaluation/test_set.pkl")
TOP_K = 5


def _load_test_set():
    with open(TEST_SET_PATH, "rb") as f:
        return pickle.load(f)


def _precision_recall_mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> tuple[float, float, float]:
    retrieved_set = set(retrieved_ids)
    hits = retrieved_set & relevant_ids

    precision = len(hits) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(hits) / len(relevant_ids) if relevant_ids else 0.0

    rr = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            rr = 1 / rank
            break

    return precision, recall, rr


def evaluate_dense_only(questions, use_hnsw: bool, top_k: int = TOP_K) -> dict:
    precisions, recalls, mrrs = [], [], []

    for q in questions:
        results = search_legal_faiss(q.question, top_k=top_k, use_hnsw=use_hnsw)
        retrieved_ids = [chunk.chunk_id for chunk, _ in results]
        p, r, rr = _precision_recall_mrr(retrieved_ids, set(q.relevant_chunk_ids))
        precisions.append(p)
        recalls.append(r)
        mrrs.append(rr)

    n = len(questions)
    return {
        "avg_precision@k": sum(precisions) / n,
        "avg_recall@k": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
    }


def evaluate_full_pipeline(questions, use_hnsw: bool, top_k: int = TOP_K, filter_to_query_document: bool = False) -> dict:
    precisions, recalls, mrrs = [], [], []

    for q in questions:
        results = retrieve_legal(q.question, top_k_final=top_k, use_hnsw=use_hnsw,filter_to_query_document=filter_to_query_document)
        retrieved_ids = []
        for r in results:
            # retrieve_legal returns dicts with char_start/char_end/source, not chunk_id directly —
            # reconstruct chunk_id lookup via source+char_start as a unique key match instead
            retrieved_ids.append(f"{r['source']}::{r['char_start']}")
        # relevant_ids need the same key shape for comparison — handled by caller
        retrieved_ids_raw = results
        p, r_, rr = _compare_by_span(retrieved_ids_raw, q.relevant_chunk_ids, q)
        precisions.append(p)
        recalls.append(r_)
        mrrs.append(rr)

    n = len(questions)
    return {
        "avg_precision@k": sum(precisions) / n,
        "avg_recall@k": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
    }


def _compare_by_span(retrieved_chunks: list[dict], relevant_chunk_ids: list[str], q) -> tuple[float, float, float]:
    relevant_set = set(relevant_chunk_ids)
    hit_count = 0
    first_hit_rank = None

    for rank, r in enumerate(retrieved_chunks, start=1):
        cid = _chunks_by_span.get((r["source"], r["char_start"], r["char_end"]))
        if cid is None:
            print(f"DEBUG: no chunk_id found for source={r['source']!r}, char_start={r['char_start']}, char_end={r['char_end']}")
        elif cid in relevant_set:
            hit_count += 1
            if first_hit_rank is None:
                first_hit_rank = rank

    precision = hit_count / len(retrieved_chunks) if retrieved_chunks else 0.0
    recall = hit_count / len(relevant_set) if relevant_set else 0.0
    rr = 1 / first_hit_rank if first_hit_rank else 0.0

    return precision, recall, rr

def _build_span_lookup():
    """Builds (source, char_start, char_end) -> chunk_id, used by _compare_by_span."""
    from backend.services.legal_corpus import _legal_store
    return {
        (c.file_path, c.char_start, c.char_end): c.chunk_id
        for c in _legal_store["chunks"]
    }


if __name__ == "__main__":
    initialize_legal_corpus()
    _chunks_by_span = _build_span_lookup()
    questions = _load_test_set()
    logger.info(f"Evaluating retrieval on {len(questions)} questions, top_k={TOP_K}")

    print("\n=== DENSE-ONLY (isolates index type) ===")
    flat_dense = evaluate_dense_only(questions, use_hnsw=False)
    hnsw_dense = evaluate_dense_only(questions, use_hnsw=True)
    print(f"Flat : {flat_dense}")
    print(f"HNSW : {hnsw_dense}")

    # Full pipeline involves reranking (slow on CPU) — use a smaller, still-credible subset
    FULL_PIPELINE_SAMPLE = 40
    sample_questions = questions[:FULL_PIPELINE_SAMPLE]

    print(f"\n=== FULL PIPELINE, no filtering (sample of {FULL_PIPELINE_SAMPLE}) ===")
    flat_full = evaluate_full_pipeline(sample_questions, use_hnsw=False)
    print(f"Flat : {flat_full}")

    print(f"\n=== FULL PIPELINE, WITH document filtering (sample of {FULL_PIPELINE_SAMPLE}) ===")
    flat_full_filtered = evaluate_full_pipeline(sample_questions, use_hnsw=False, filter_to_query_document=True)
    print(f"Flat : {flat_full_filtered}")