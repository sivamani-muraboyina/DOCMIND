from __future__ import annotations
import re, string
from typing import Dict, List, Tuple

import nltk
from loguru import logger
from rank_bm25 import BM25Okapi

from backend.utils.chunker import Chunk

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
_STOPWORDS = set(stopwords.words("english"))

# session_id → {"bm25": BM25Okapi, "chunks": List[Chunk]}
_store: Dict[str, dict] = {}


def _tokenize(text: str) -> List[str]:
    text   = text.lower()
    text   = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def build_bm25_index(session_id: str, chunks: List[Chunk]) -> int:
    pairs = [
        (_tokenize(c.content), c)
        for c in chunks
        if _tokenize(c.content)
    ]
    if not pairs:
        logger.warning(f"BM25: empty corpus for session {session_id}")
        return 0

    tokens_list, valid_chunks = zip(*pairs)
    _store[session_id] = {
        "bm25":   BM25Okapi(list(tokens_list)),
        "chunks": list(valid_chunks),
    }
    logger.info(f"BM25 index ready | session={session_id} | docs={len(valid_chunks)}")
    return len(valid_chunks)


def search_bm25(session_id: str, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
    if session_id not in _store:
        raise KeyError(f"Session not found: {session_id}")

    bm25   = _store[session_id]["bm25"]
    chunks = _store[session_id]["chunks"]
    qtoks  = _tokenize(query)

    if not qtoks:
        return []

    scores  = bm25.get_scores(qtoks)
    top_n   = min(top_k, len(chunks))
    indices = scores.argsort()[::-1][:top_n]

    return [
        (chunks[i], float(scores[i]))
        for i in indices
        if scores[i] > 0
    ]


def bm25_session_exists(session_id: str) -> bool:
    return session_id in _store


def delete_bm25_session(session_id: str) -> bool:
    return bool(_store.pop(session_id, None))
