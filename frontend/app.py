"""
DocMind — Streamlit frontend for the Hybrid RAG chatbot.

Run with:
    streamlit run frontend/app.py
"""
import requests
import streamlit as st

BACKEND = "http://localhost:8000"

st.set_page_config(
    page_title="DocMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    for k, v in {
        "session_id":  None,
        "filename":    None,
        "doc_meta":    {},
        "messages":    [],
        "ready":       False,
    }.items():
        st.session_state.setdefault(k, v)

_init()


# ── API helpers ───────────────────────────────────────────────────────────────
def backend_alive() -> bool:
    try:
        return requests.get(f"{BACKEND}/health", timeout=3).status_code == 200
    except Exception:
        return False


def api_upload(data: bytes, name: str) -> dict | None:
    try:
        r = requests.post(
            f"{BACKEND}/upload/",
            files={"file": (name, data, "application/pdf")},
            timeout=120,
        )
        if r.status_code == 201:
            return r.json()
        st.error(f"Upload failed: {r.json().get('detail', r.status_code)}")
    except requests.ConnectionError:
        st.error("Cannot reach backend on port 8000.")
    except Exception as e:
        st.error(f"Upload error: {e}")
    return None


def api_ask(question: str) -> dict | None:
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-10:]
    ]
    try:
        r = requests.post(
            f"{BACKEND}/ask/",
            json={
                "session_id":   st.session_state.session_id,
                "question":     question,
                "chat_history": history,
                "top_k":        5,
            },
            timeout=120,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"Error: {r.json().get('detail', r.status_code)}")
    except requests.Timeout:
        st.error("Request timed out — the LLM may be slow, try again.")
    except requests.ConnectionError:
        st.error("Lost connection to backend.")
    except Exception as e:
        st.error(f"Error: {e}")
    return None


# ── Render helpers ────────────────────────────────────────────────────────────
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


def render_stats():
    meta = st.session_state.doc_meta
    if not meta:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pages",    meta.get("num_pages",      "—"))
    c2.metric("Chunks",   meta.get("num_chunks",     "—"))
    c3.metric("Doc Type", meta.get("doc_type",       "—"))
    c4.metric("Strategy", meta.get("chunk_strategy", "—"))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 DocMind")
    st.caption("Hybrid RAG Chatbot")
    st.divider()

    alive = backend_alive()
    if alive:
        st.success("API connected", icon="✅")
    else:
        st.error("API offline — start the FastAPI backend on port 8000", icon="🔴")

    st.markdown("#### 📂 Upload PDF")
    pdf = st.file_uploader("Drop your PDF here", type=["pdf"])

    if pdf and not st.session_state.ready:
        if st.button("⚡ Index Document", use_container_width=True):
            with st.spinner("Processing…"):
                result = api_upload(pdf.read(), pdf.name)
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
    render_stats()
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
                        res = api_ask(sug)
                    if res:
                        st.session_state.messages.append({
                            "role":    "assistant",
                            "content": res.get("answer", ""),
                            "sources": res.get("retrieved_chunks", []),
                        })
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
                res = api_ask(q)
            if res:
                answer  = res.get("answer", "")
                sources = res.get("retrieved_chunks", [])
                st.write(answer)
                render_sources(sources)
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "sources": sources,
                })
