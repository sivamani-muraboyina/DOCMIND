from __future__ import annotations
from typing import Dict, List, Tuple

import faiss
import numpy as np
from loguru import logger

from backend.services.embeddings import embed_texts, embed_query
from backend.utils.chunker import Chunk

# session_id → {"index": faiss.Index, "chunks": List[Chunk]}
_store: Dict[str, dict] = {}


def build_faiss_index(session_id: str, chunks: List[Chunk]) -> int:
    texts      = [c.content for c in chunks]
    embeddings = embed_texts(texts)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    _store[session_id] = {"index": index, "chunks": chunks}
    logger.info(f"FAISS index ready | session={session_id} | vectors={index.ntotal}")
    return index.ntotal


def search_faiss(session_id: str, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
    if session_id not in _store:
        raise KeyError(f"Session not found: {session_id}")

    index  = _store[session_id]["index"]
    chunks = _store[session_id]["chunks"]

    q_vec         = embed_query(query)
    scores, idxs  = index.search(q_vec, min(top_k, index.ntotal))

    return [
        (chunks[i], float(s))
        for s, i in zip(scores[0], idxs[0])
        if i != -1
    ]


def session_exists(session_id: str) -> bool:
    return session_id in _store


def get_active_sessions() -> int:
    return len(_store)


def delete_session(session_id: str) -> bool:
    return bool(_store.pop(session_id, None))
