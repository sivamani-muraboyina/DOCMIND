from __future__ import annotations
import os

from fastapi import APIRouter
from loguru import logger

from backend.models.schemas import AskLegalRequest, AskLegalResponse, LegalRetrievedChunk
from backend.services.llm_chain import generate_answer
from backend.services.retriever import retrieve_legal

router = APIRouter(prefix="/ask-legal", tags=["Legal Corpus"])

TOP_K_DENSE  = int(os.getenv("TOP_K_DENSE",  "10"))
TOP_K_SPARSE = int(os.getenv("TOP_K_SPARSE", "10"))
ALPHA        = float(os.getenv("HYBRID_ALPHA", "0.6"))
LLM_TEMP     = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
LLM_TOKENS   = int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "2048"))


@router.post("/", response_model=AskLegalResponse)
async def ask_legal(req: AskLegalRequest) -> AskLegalResponse:
    logger.info(f"Legal query | use_hnsw={req.use_hnsw} | q='{req.question[:60]}'")

    chunks = retrieve_legal(
        query        = req.question,
        top_k_dense  = TOP_K_DENSE,
        top_k_sparse = TOP_K_SPARSE,
        top_k_final  = req.top_k,
        alpha        = ALPHA,
        use_hnsw     = req.use_hnsw,
    )

    if not chunks:
        return AskLegalResponse(
            answer="I couldn't find relevant content in the legal corpus.",
            question=req.question,
            retrieved_chunks=[],
            model_used="llama-3.1-8b-instant",
            index_used="hnsw" if req.use_hnsw else "flat",
        )

    answer, model = generate_answer(
        question    = req.question,
        chunks      = chunks,
        history     = [],
        temperature = LLM_TEMP,
        max_tokens  = LLM_TOKENS,
    )

    return AskLegalResponse(
        answer           = answer,
        question         = req.question,
        retrieved_chunks = [LegalRetrievedChunk(**c) for c in chunks],
        model_used       = model,
        index_used       = "hnsw" if req.use_hnsw else "flat",
    )