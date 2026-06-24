# 🧠 DocMind — Hybrid RAG Document Chatbot

> Two-mode intelligent document analysis — chat with any PDF or query a legal contract corpus using state-of-the-art hybrid retrieval powered by Groq LLaMA.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![LangChain](https://img.shields.io/badge/LangChain-0.2-yellow)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.1%208B-orange)
[![Two-Mode Demo](https://img.shields.io/badge/🚀%20Full%20Demo%20(2%20Modes)-Render-46E3B7?style=flat)](https://docmind-by-siva.onrender.com)
[![Single-Mode Demo](https://img.shields.io/badge/⚡%20Quick%20Demo%20(Doc%20Mode)-Streamlit-FF4B4B?style=flat)](https://docmind-by-siva.streamlit.app)

---

## 🚀 Live Demos

| Version | Link | Modes Available |
|---|---|---|
| **Full (Two-Mode)** | [https://docmind-by-siva.onrender.com](https://docmind-by-siva.onrender.com) | Document Q&A + Legal Contract Analysis |
| **Quick Demo** | [https://docmind-by-siva.streamlit.app](https://docmind-by-siva.streamlit.app) | Document Q&A only |

> ⚠️ Full demo hosted on Render free tier — may take ~30s to wake up on first load.

---

## 👔 For Recruiters — Try It in 60 Seconds

**Mode 1 — Document Q&A:**
1. Open the [Full Demo](https://docmind-by-siva.onrender.com)
2. Upload any PDF (resume, research paper, report)
3. Ask *"Summarize the key points"* or *"What is the methodology?"*
4. See hybrid retrieval return answers with page-level citations

**Mode 2 — Legal Contract Analysis:**
1. Switch to **Legal Mode** in the sidebar
2. Ask contract-related questions like *"What are the termination clauses?"* or *"Summarize indemnification obligations"*
3. System queries a pre-indexed CUAD legal corpus — no upload needed

> **What this demonstrates:** end-to-end LLM application development — PDF parsing, vector indexing, hybrid retrieval, reranking, multi-mode architecture, prompt engineering, and Docker deployment. Built and deployed solo.

---

## 📸 App Preview

### Home Screen
![DocMind Home](docs/screenshot_home.png)

### Chat in Action
![DocMind Chat](docs/screenshot_chat.png)

---

## ✨ Features

### 📄 Mode 1 — Document Q&A
- **PDF Upload** — Upload any PDF up to 50MB
- **Adaptive Chunking** — Auto-classifies PDF as structured / unstructured / mixed
- **Hybrid Retrieval** — FAISS dense search + BM25 sparse search via Reciprocal Rank Fusion
- **Cross-Encoder Reranking** — ms-marco-MiniLM for maximum precision
- **Source Citations** — Every answer cites exact chunks and page numbers
- **Chat History** — Multi-turn conversation with memory (last 6 turns)

### ⚖️ Mode 2 — Legal Contract Analysis
- **Pre-indexed Corpus** — CUAD (Contract Understanding Atticus Dataset) indexed at startup
- **Dual FAISS Index** — Flat (exact) + HNSW (approximate) for retrieval benchmarking
- **Legal-aware Chunking** — Clause-boundary-respecting chunker for contract documents
- **No Upload Needed** — Instant answers over the legal corpus out of the box

### ⚡ Shared Infrastructure
- **Groq LLaMA 3.1 8B Instant** — Fast, grounded answers strictly from retrieved context
- **sentence-transformers all-MiniLM-L6-v2** — Embeddings for both modes
- **FastAPI backend + Streamlit frontend** — Clean separation via REST API
- **Docker deployment** — Single container, production-ready

---

## 🏗️ Architecture

### System Overview
![System Architecture](docs/architecture.svg)

### Upload & Indexing Pipeline
![Upload Flow](docs/upload_flow.svg)

### Query & Answer Pipeline
![Query Flow](docs/query_flow.svg)

---

## 🚀 Quick Start (Local)

### 1. Clone the repo
```bash
git clone https://github.com/sivamani-muraboyina/DOCMIND.git
cd DOCMIND
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

Open `.env` and add your Groq API key — get one free at [console.groq.com](https://console.groq.com):
```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run Backend (Terminal 1)
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Run Frontend (Terminal 2)
```bash
streamlit run frontend/app.py
```

### 7. Open in browser
```
http://localhost:8501
```

---

## 🐳 Run with Docker

```bash
copy .env.example .env   # fill in your GROQ_API_KEY
docker-compose up --build
```

| Service | URL |
|---|---|
| Frontend (Streamlit) | http://localhost:8501 |
| Backend (FastAPI) | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |

---

## 📁 Project Structure

```
DOCMIND/
├── backend/
│   ├── models/
│   │   └── schemas.py              # Pydantic request/response models
│   ├── routers/
│   │   ├── upload.py               # POST /upload endpoint
│   │   ├── chat.py                 # POST /ask endpoint (Document mode)
│   │   └── legal.py                # POST /ask-legal endpoint (Legal mode)
│   ├── services/
│   │   ├── llm_chain.py            # Groq LLaMA via LangChain
│   │   ├── embeddings.py           # sentence-transformers (shared)
│   │   ├── vector_store.py         # FAISS in-memory index (Document mode)
│   │   ├── bm25_store.py           # BM25 in-memory index (Document mode)
│   │   ├── retriever.py            # Hybrid retrieval + cross-encoder reranking
│   │   ├── pdf_processor.py        # PDF text extraction
│   │   └── legal_corpus.py         # CUAD corpus indexing + retrieval (Legal mode)
│   └── utils/
│       ├── chunker.py              # Adaptive chunking (Document mode)
│       ├── doc_classifier.py       # PDF type classification
│       └── legal_chunker.py        # Clause-aware chunking (Legal mode)
├── frontend/
│   └── app.py                      # Streamlit UI (two-mode switcher)
├── data/
│   └── raw/corpus/cuad/            # Legal corpus .txt files (not committed)
├── docs/                           # Architecture diagrams + screenshots
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Get free at [console.groq.com](https://console.groq.com) |
| `UPLOAD_DIR` | `./uploads` | Where uploaded PDFs are saved |
| `MAX_FILE_SIZE_MB` | `50` | Max upload size in MB |
| `TOP_K_DENSE` | `10` | FAISS top-k results |
| `TOP_K_SPARSE` | `10` | BM25 top-k results |
| `TOP_K_RERANK` | `5` | Final chunks after reranking |
| `HYBRID_ALPHA` | `0.6` | Dense vs sparse weight (1.0 = all dense) |
| `GROQ_TEMPERATURE` | `0.2` | LLM temperature |
| `GROQ_MAX_OUTPUT_TOKENS` | `2048` | Max response tokens |
| `CHUNK_SIZE_DEFAULT` | `512` | Default chunk size |
| `CHUNK_SIZE_STRUCTURED` | `256` | Chunk size for structured PDFs |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload/` | Upload and index a PDF — returns `session_id` |
| `POST` | `/ask/` | Ask a question in Document mode using `session_id` |
| `POST` | `/ask-legal/` | Ask a question in Legal mode (no session needed) |
| `GET` | `/health` | API status + active session count |
| `GET` | `/docs` | Swagger UI |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq — LLaMA 3.1 8B Instant |
| **Embeddings** | sentence-transformers all-MiniLM-L6-v2 |
| **Dense Search** | FAISS IndexFlatIP + HNSW |
| **Sparse Search** | BM25 Okapi |
| **Reranker** | cross-encoder ms-marco-MiniLM-L-6-v2 |
| **Legal Corpus** | CUAD (Contract Understanding Atticus Dataset) |
| **PDF Extraction** | pdfplumber + PyMuPDF |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Orchestration** | LangChain |
| **Deployment** | Docker on Render |

---

## 🙏 Acknowledgements

- [Groq](https://groq.com) for blazing fast LLM inference
- [LangChain](https://langchain.com) for LLM orchestration
- [Hugging Face](https://huggingface.co) for embedding and reranker models
- [Streamlit](https://streamlit.io) for the UI framework
- [FAISS](https://github.com/facebookresearch/faiss) by Meta AI for vector search
- [CUAD](https://www.atticusprojectai.org/cuad) for the legal contract dataset

---

## 📬 Contact

**Sivamani Muraboyina**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sivamani-muraboyina-5873b52a7/)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:sivamani.m23@iiits.in)
