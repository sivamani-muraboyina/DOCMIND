"""
Hugging Face Spaces entry point.

HF Spaces runs a single process — we launch FastAPI in a background thread
and Streamlit as the main process. The Streamlit UI talks to the FastAPI
backend on localhost:8000.

Required HF Spaces secrets (set in repo Settings → Secrets):
  GROQ_API_KEY
"""
import threading
import uvicorn

def _run_backend():
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, log_level="warning")

# Start FastAPI in a background thread
t = threading.Thread(target=_run_backend, daemon=True)
t.start()

# Streamlit is started by HF Spaces via:  streamlit run app.py
# The import below makes this file importable as a module too.
import frontend.app  # noqa: F401 — triggers st.set_page_config etc.
