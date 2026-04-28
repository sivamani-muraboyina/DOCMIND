from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    STRUCTURED   = "structured"
    UNSTRUCTURED = "unstructured"
    MIXED        = "mixed"


class ChunkStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    RECURSIVE  = "recursive"
    SEMANTIC   = "semantic"


class UploadResponse(BaseModel):
    session_id:     str
    filename:       str
    num_pages:      int
    num_chunks:     int
    doc_type:       DocumentType
    chunk_strategy: ChunkStrategy
    chunk_size:     int
    chunk_overlap:  int
    message:        str = "Document indexed successfully"


class ChatMessage(BaseModel):
    role:    str
    content: str


class AskRequest(BaseModel):
    session_id:   str
    question:     str = Field(..., min_length=1, max_length=2000)
    chat_history: List[ChatMessage] = Field(default_factory=list)
    top_k:        int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    content:          str
    source:           str
    page:             Optional[int] = None
    score:            float
    retrieval_method: str


class AskResponse(BaseModel):
    answer:           str
    session_id:       str
    question:         str
    retrieved_chunks: List[RetrievedChunk]
    model_used:       str
    tokens_used:      Optional[int] = None


class HealthResponse(BaseModel):
    status:          str = "ok"
    version:         str = "1.0.0"
    active_sessions: int = 0
