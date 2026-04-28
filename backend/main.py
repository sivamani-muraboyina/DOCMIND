from __future__ import annotations
import os, time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

load_dotenv()

from backend.models.schemas import HealthResponse
from backend.routers import chat, upload
from backend.services.embeddings import load_embedding_model
from backend.services.retriever import load_reranker
from backend.services.vector_store import get_active_sessions


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(os.getenv("UPLOAD_DIR", "./uploads")).mkdir(parents=True, exist_ok=True)
    logger.info("Pre-loading models...")
    load_embedding_model()
    load_reranker()
    logger.info("Models ready — API is live.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title       = "DocMind API",
    description = "Hybrid RAG (FAISS + BM25 + Cross-encoder) over uploaded PDFs",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    t = time.time()
    resp = await call_next(request)
    resp.headers["X-Response-Time-Ms"] = str(round((time.time() - t) * 1000, 2))
    return resp


@app.exception_handler(Exception)
async def catch_all(request: Request, exc: Exception):
    logger.error(f"Unhandled error at {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


app.include_router(upload.router)
app.include_router(chat.router)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(status="ok", version="1.0.0", active_sessions=get_active_sessions())


@app.get("/", tags=["System"])
async def root():
    return {"message": "DocMind API", "docs": "/docs", "health": "/health"}
