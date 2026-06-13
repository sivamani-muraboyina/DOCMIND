"""
DocMind — Standalone Streamlit app (no separate backend needed).
All RAG logic is embedded directly.
"""
from __future__ import annotations

import os
import re
import string
import uuid
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
import streamlit as st

# ── Download NLTK data ────────────────────────────────────────────────────────
for pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords
_STOPWORDS = set(stopwords.words("english"))

# ── Enums ─────────────────────────────────────────────────────────────────────
class DocumentType(str, Enum):
    STRUCTURED   = "structured"
    UNSTRUCTURED = "unstructured"
    MIXED        = "mixed"

class ChunkStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    RECURSIVE  = "recursive"
    SEMANTIC   = "semantic"

# ── Chunk dataclass ───────────────────────────────────────────────────────────
@dataclass
class Chunk:
    content:  str
    metadata: dict

# ── Embedding model ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: List[str]) -> np.ndarray:
    model = load_embedding_model()
    return model.encode(
        texts, batch_size=64, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype(np.float32)

def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])

# ── Cross-encoder reranker ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading reranker…")
def load_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

# ── In-memory stores ──────────────────────────────────────────────────────────
_faiss_store: Dict[str, dict] = {}
_bm25_store:  Dict[str, dict] = {}

# ── FAISS ─────────────────────────────────────────────────────────────────────
def build_faiss_index(session_id: str, chunks: List[Chunk]) -> int:
    import faiss
    texts      = [c.content for c in chunks]
    embeddings = embed_texts(texts)
    dim        = embeddings.shape[1]
    index      = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    _faiss_store[session_id] = {"index": index, "chunks": chunks}
    return index.ntotal

def search_faiss(session_id: str, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
    import faiss
    entry  = _faiss_store[session_id]
    index  = entry["index"]
    chunks = entry["chunks"]
    q_vec  = embed_query(query)
    scores, idxs = index.search(q_vec, min(top_k, index.ntotal))
    return [(chunks[i], float(s)) for s, i in zip(scores[0], idxs[0]) if i != -1]

# ── BM25 ──────────────────────────────────────────────────────────────────────
def _tokenize(text: str) -> List[str]:
    text   = text.lower()
    text   = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

def build_bm25_index(session_id: str, chunks: List[Chunk]) -> int:
    from rank_bm25 import BM25Okapi
    pairs = [(_tokenize(c.content), c) for c in chunks if _tokenize(c.content)]
    if not pairs:
        return 0
    tokens_list, valid_chunks = zip(*pairs)
    _bm25_store[session_id] = {
        "bm25":   BM25Okapi(list(tokens_list)),
        "chunks": list(valid_chunks),
    }
    return len(valid_chunks)

def search_bm25(session_id: str, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
    entry  = _bm25_store[session_id]
    bm25   = entry["bm25"]
    chunks = entry["chunks"]
    qtoks  = _tokenize(query)
    if not qtoks:
        return []
    scores  = bm25.get_scores(qtoks)
    top_n   = min(top_k, len(chunks))
    indices = scores.argsort()[::-1][:top_n]
    return [(chunks[i], float(scores[i])) for i in indices if scores[i] > 0]

# ── PDF processing ────────────────────────────────────────────────────────────
def classify_pdf(pdf_path: Path) -> DocumentType:
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            all_lines, table_pages = [], 0
            for page in pdf.pages:
                tables = page.extract_tables() or []
                if any(len(t) > 1 for t in tables):
                    table_pages += 1
                text  = page.extract_text() or ""
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                all_lines.extend(lines)
            total       = len(pdf.pages)
            table_ratio = table_pages / max(total, 1)
            short_ratio = len([l for l in all_lines if len(l) < 50]) / max(len(all_lines), 1)
            avg_len     = sum(len(l) for l in all_lines) / max(len(all_lines), 1)
            if table_ratio >= 0.15 or short_ratio >= 0.40:
                return DocumentType.STRUCTURED
            elif short_ratio < 0.20 and avg_len > 80:
                return DocumentType.UNSTRUCTURED
            else:
                return DocumentType.MIXED
    except Exception:
        return DocumentType.MIXED

def extract_pages(pdf_path: Path, doc_type: DocumentType) -> Tuple[List[dict], int]:
    source = pdf_path.name
    try:
        if doc_type == DocumentType.STRUCTURED:
            import pdfplumber
            pages = []
            with pdfplumber.open(pdf_path) as pdf:
                for n, page in enumerate(pdf.pages, 1):
                    parts = []
                    for table in (page.extract_tables() or []):
                        rows = [" | ".join(str(c or "").strip() for c in row) for row in table]
                        if rows:
                            parts.append("\n".join(rows))
                    raw = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                    if raw.strip():
                        parts.append(raw)
                    pages.append({"text": "\n\n".join(parts), "page": n, "source": source})
            return pages, len(pages)
        else:
            import fitz
            pages = []
            doc   = fitz.open(str(pdf_path))
            for n, page in enumerate(doc, 1):
                blocks = page.get_text("blocks", sort=True)
                text   = "\n\n".join(b[4].strip() for b in blocks if b[6] == 0 and b[4].strip())
                pages.append({"text": text, "page": n, "source": source})
            doc.close()
            return pages, len(pages)
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return [], 0

# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_document(pages, doc_type, chunk_size=512, chunk_overlap=100) -> Tuple[List[Chunk], ChunkStrategy]:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    strategy_map = {
        DocumentType.STRUCTURED:   ChunkStrategy.FIXED_SIZE,
        DocumentType.UNSTRUCTURED: ChunkStrategy.RECURSIVE,
        DocumentType.MIXED:        ChunkStrategy.SEMANTIC,
    }
    strategy = strategy_map[doc_type]

    if strategy == ChunkStrategy.SEMANTIC:
        chunks, overlap_n = [], max(1, chunk_overlap // 80)
        for p in pages:
            text = p["text"].strip()
            if not text:
                continue
            try:
                sentences = nltk.sent_tokenize(text)
            except Exception:
                sentences = re.split(r"(?<=[.!?])\s+", text)
            current, cur_len, idx = [], 0, 0
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if cur_len + len(sent) > chunk_size and current:
                    chunks.append(Chunk(
                        content=" ".join(current),
                        metadata={"source": p.get("source","doc"), "page": p.get("page"), "chunk_index": idx, "strategy": strategy.value}
                    ))
                    idx   += 1
                    current = current[-overlap_n:]
                    cur_len = sum(len(s) for s in current)
                current.append(sent)
                cur_len += len(sent)
            if current:
                chunks.append(Chunk(
                    content=" ".join(current),
                    metadata={"source": p.get("source","doc"), "page": p.get("page"), "chunk_index": idx, "strategy": strategy.value}
                ))
        return chunks, strategy

    seps = {
        ChunkStrategy.FIXED_SIZE: ["\n\n", "\n", " | ", "|", " ", ""],
        ChunkStrategy.RECURSIVE:  ["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""],
    }
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=seps[strategy], length_function=len,
    )
    chunks, idx = [], 0
    for p in pages:
        text = p["text"].strip()
        if not text:
            continue
        for split in splitter.split_text(text):
            split = split.strip()
            if split:
                chunks.append(Chunk(
                    content=split,
                    metadata={"source": p.get("source","doc"), "page": p.get("page"), "chunk_index": idx, "strategy": strategy.value}
                ))
                idx += 1
    return chunks, strategy

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(session_id: str, query: str, top_k_final: int = 5) -> List[dict]:
    dense_hits, sparse_hits = [], []
    try:
        dense_hits  = search_faiss(session_id, query, top_k=10)
    except Exception:
        pass
    try:
        sparse_hits = search_bm25(session_id, query, top_k=10)
    except Exception:
        pass
    if not dense_hits and not sparse_hits:
        return []

    # Reciprocal Rank Fusion
    k           = 60
    alpha       = 0.6
    dense_rank  = {c.content: r + 1 for r, (c, _) in enumerate(dense_hits)}
    sparse_rank = {c.content: r + 1 for r, (c, _) in enumerate(sparse_hits)}
    chunk_map   = {c.content: c for c, _ in dense_hits + sparse_hits}

    scored = []
    for content in set(dense_rank) | set(sparse_rank):
        dr = dense_rank.get(content)
        sr = sparse_rank.get(content)
        score  = (alpha * (1 / (k + dr)) if dr else 0.0) + ((1 - alpha) * (1 / (k + sr)) if sr else 0.0)
        method = "hybrid" if (dr and sr) else ("dense" if dr else "sparse")
        scored.append((chunk_map[content], score, method))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Rerank
    if len(scored) > 1:
        reranker = load_reranker()
        pairs    = [(query, chunk.content) for chunk, _, _ in scored]
        scores   = reranker.predict(pairs, show_progress_bar=False)
        scored   = [(chunk, float(s), method) for (chunk, _, method), s in zip(scored, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)

    return [
        {
            "content":          chunk.content,
            "source":           chunk.metadata.get("source", "unknown"),
            "page":             chunk.metadata.get("page"),
            "score":            round(score, 4),
            "retrieval_method": method,
        }
        for chunk, score, method in scored[:top_k_final]
    ]

# ── LLM ───────────────────────────────────────────────────────────────────────
def generate_answer(question: str, chunks: List[dict], history: List[dict]) -> str:
    from langchain_groq import ChatGroq
    from langchain.schema import AIMessage, HumanMessage, SystemMessage

    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("GROQ_API_KEY is not set. Add it in Streamlit Cloud → Settings → Secrets.")
        return ""

    context_parts = []
    for i, c in enumerate(chunks, 1):
        loc = f"[{c.get('source','doc')} · p.{c['page']}]" if c.get("page") else f"[{c.get('source','doc')}]"
        context_parts.append(f"[Excerpt {i}] {loc}\n{c['content']}")
    context = "\n\n".join(context_parts)

    system = f"""You are a precise document analyst. Answer questions ONLY from the context below.
Rules:
- Use only what is in the provided context.
- If the answer is not in the context, say so explicitly.
- Cite page numbers when they help.
- Use bullet points for multi-part answers.

Document context:
{context}"""

    llm      = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, max_tokens=1024, groq_api_key=api_key)
    messages = [SystemMessage(content=system)]
    for msg in (history or [])[-6:]:
        cls = HumanMessage if msg["role"] == "user" else AIMessage
        messages.append(cls(content=msg["content"]))
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return response.content.strip()

# ── Process uploaded PDF ──────────────────────────────────────────────────────
def process_pdf(pdf_bytes: bytes, filename: str) -> Optional[dict]:
    upload_dir = Path("/tmp/docmind_uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    session_id = str(uuid.uuid4())
    save_path  = upload_dir / f"{session_id}_{filename}"
    save_path.write_bytes(pdf_bytes)

    with st.spinner("Classifying document…"):
        doc_type = classify_pdf(save_path)

    chunk_sizes    = {DocumentType.STRUCTURED: 256, DocumentType.UNSTRUCTURED: 512, DocumentType.MIXED: 512}
    chunk_overlaps = {DocumentType.STRUCTURED: 50,  DocumentType.UNSTRUCTURED: 100, DocumentType.MIXED: 100}
    chunk_size    = chunk_sizes[doc_type]
    chunk_overlap = chunk_overlaps[doc_type]

    with st.spinner("Extracting text…"):
        pages, num_pages = extract_pages(save_path, doc_type)

    if not any(p["text"].strip() for p in pages):
        st.error("No extractable text found in this PDF.")
        return None

    with st.spinner("Chunking document…"):
        chunks, strategy = chunk_document(pages, doc_type, chunk_size, chunk_overlap)

    if not chunks:
        st.error("Document produced zero chunks.")
        return None

    with st.spinner("Building FAISS index…"):
        n_indexed = build_faiss_index(session_id, chunks)

    with st.spinner("Building BM25 index…"):
        try:
            build_bm25_index(session_id, chunks)
        except Exception:
            pass

    return {
        "session_id":     session_id,
        "filename":       filename,
        "num_pages":      num_pages,
        "num_chunks":     n_indexed,
        "doc_type":       doc_type.value,
        "chunk_strategy": strategy.value,
        "chunk_size":     chunk_size,
        "chunk_overlap":  chunk_overlap,
    }

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="DocMind", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

def _init():
    for k, v in {
        "session_id": None, "filename": None,
        "doc_meta": {}, "messages": [], "ready": False,
    }.items():
        st.session_state.setdefault(k, v)

_init()

def render_sources(sources: list):
    if not sources:
        return
    with st.expander(f"📎 {len(sources)} retrieved chunks"):
        for i, s in enumerate(sources, 1):
            method = s.get("retrieval_method", "unknown")
            st.caption(f"#{i} · {s.get('source','doc')} · page {s.get('page','?')} · score {s.get('score',0):.4f} · [{method}]")
            st.text(s.get("content", "")[:420])
            if i < len(sources):
                st.divider()

def render_messages():
    for m in st.session_state.messages:
        role = "user" if m["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.write(m["content"])
            if role == "assistant" and m.get("sources"):
                render_sources(m["sources"])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 DocMind")
    st.caption("Hybrid RAG Chatbot")
    st.divider()

    st.markdown("#### 📂 Upload PDF")
    pdf = st.file_uploader("Drop your PDF here", type=["pdf"])

    if pdf and not st.session_state.ready:
        if st.button("⚡ Index Document", use_container_width=True):
            result = process_pdf(pdf.read(), pdf.name)
            if result:
                st.session_state.update({
                    "session_id": result["session_id"],
                    "filename":   result["filename"],
                    "doc_meta":   result,
                    "messages":   [],
                    "ready":      True,
                })
                st.success("Ready to chat!")
                st.rerun()

    elif pdf and st.session_state.ready:
        if st.button("🔄 New Document", use_container_width=True):
            st.session_state.update({"ready": False, "session_id": None, "doc_meta": {}, "messages": []})
            st.rerun()

    if st.session_state.session_id:
        st.divider()
        st.markdown("#### 📋 Session")
        meta = st.session_state.doc_meta
        st.write(f"**File:** {st.session_state.filename}")
        st.caption(f"ID: {st.session_state.session_id[:20]}…")
        st.write(f"**Pages:** {meta.get('num_pages','?')}  |  **Chunks:** {meta.get('num_chunks','?')}")
        st.write(f"**Type:** {meta.get('doc_type','?')}  |  **Strategy:** {meta.get('chunk_strategy','?')}")

    if st.session_state.messages:
        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    with st.expander("⚙️ Pipeline"):
        st.markdown("""
**Upload flow**
1. Classify PDF → structured / unstructured / mixed
2. Extract text (pdfplumber or PyMuPDF)
3. Chunk with adaptive strategy
4. Build FAISS index (dense)
5. Build BM25 index (sparse)

**Query flow**
1. Dense search → FAISS cosine similarity
2. Sparse search → BM25 keyword match
3. Fuse scores → Reciprocal Rank Fusion
4. Rerank → cross-encoder (ms-marco-MiniLM)
5. Generate → Groq (LLaMA 3.1 8B)
""")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🧠 DocMind")
st.caption("Hybrid RAG · Dense + Sparse Retrieval · Cross-Encoder Reranking")
st.divider()

if not st.session_state.ready:
    st.info("Upload a PDF from the sidebar to get started.", icon="📑")
    col1, col2, col3 = st.columns(3)
    col1.info("🔵 **FAISS Dense**\n\nSemantic similarity search over embedded chunks")
    col2.warning("🟡 **BM25 Sparse**\n\nKeyword matching with TF-IDF-like scoring")
    col3.success("⚡ **Cross-Encoder**\n\nReranks top candidates for maximum precision")
else:
    meta = st.session_state.doc_meta
    if meta:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pages",    meta.get("num_pages", "—"))
        c2.metric("Chunks",   meta.get("num_chunks", "—"))
        c3.metric("Doc Type", meta.get("doc_type", "—"))
        c4.metric("Strategy", meta.get("chunk_strategy", "—"))
    st.divider()

    if not st.session_state.messages:
        st.info(f"Document ready. Ask anything about **{st.session_state.filename}**", icon="💬")
        st.markdown("**Try asking:**")
        suggestions = ["Summarize key points", "What are the main conclusions?", "List findings", "What method was used?"]
        scols = st.columns(len(suggestions))
        for col, sug in zip(scols, suggestions):
            with col:
                if st.button(sug, use_container_width=True, key=f"sug_{sug}"):
                    st.session_state.messages.append({"role": "user", "content": sug, "sources": []})
                    with st.spinner("Thinking…"):
                        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]]
                        chunks  = retrieve(st.session_state.session_id, sug)
                        answer  = generate_answer(sug, chunks, history)
                    if answer:
                        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": chunks})
                    st.rerun()
    else:
        render_messages()

    user_q = st.chat_input("Ask a question about your document…")
    if user_q and user_q.strip():
        q = user_q.strip()
        st.session_state.messages.append({"role": "user", "content": q, "sources": []})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]]
                chunks  = retrieve(st.session_state.session_id, q)
                answer  = generate_answer(q, chunks, history)
            if answer:
                st.write(answer)
                render_sources(chunks)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": chunks})
