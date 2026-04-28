from __future__ import annotations
from functools import lru_cache
from typing import List

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.array([])
    model = load_embedding_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 100,
    ).astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])
