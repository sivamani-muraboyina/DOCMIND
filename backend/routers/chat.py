from __future__ import annotations
import os

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from backend.models.schemas import AskRequest, AskResponse, RetrievedChunk
from backend.services.llm_chain import generate_answer
from backend.services.retriever import retrieve
from backend.services.vector_store import session_exists

router = APIRouter(prefix="/ask", tags=["Chat"])

TOP_K_DENSE  = int(os.getenv("TOP_K_DENSE",   "10"))
TOP_K_SPARSE = int(os.getenv("TOP_K_SPARSE",  "10"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK",  "5"))
ALPHA        = float(os.getenv("HYBRID_ALPHA", "0.6"))
LLM_TEMP   = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
LLM_TOKENS = int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "2048"))


@router.post("/", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    if not session_exists(req.session_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Session '{req.session_id}' not found.")

    logger.info(f"Query | session={req.session_id[:8]} | q='{req.question[:60]}'")

    chunks = retrieve(
        session_id   = req.session_id,
        query        = req.question,
        top_k_dense  = TOP_K_DENSE,
        top_k_sparse = TOP_K_SPARSE,
        top_k_final  = req.top_k or TOP_K_RERANK,
        alpha        = ALPHA,
    )

    if not chunks:
        return AskResponse(
            answer           = "I couldn't find relevant content in the document. Try rephrasing your question.",
            session_id       = req.session_id,
            question         = req.question,
            retrieved_chunks = [],
            model_used       = "llama-3.1-8b-instant",
        )

    history = [{"role": m.role, "content": m.content} for m in req.chat_history]
    answer, model = generate_answer(
        question    = req.question,
        chunks      = chunks,
        history     = history,
        temperature = LLM_TEMP,
        max_tokens  = LLM_TOKENS,
    )

    return AskResponse(
        answer           = answer,
        session_id       = req.session_id,
        question         = req.question,
        retrieved_chunks = [RetrievedChunk(**c) for c in chunks],
        model_used       = model,
    )
