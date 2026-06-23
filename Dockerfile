
FROM python:3.11-slim
 
WORKDIR /app
 
# System deps for PyMuPDF, pdfplumber, faiss-cpu (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1-mesa-glx libgomp1 \
    && rm -rf /var/lib/apt/lists/*
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Pre-download NLTK data at build time
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
 
COPY . .
 
# HF Spaces runs as non-root — give write access
RUN mkdir -p /app/uploads /app/data && chmod -R 777 /app/uploads /app/data /app
 
# HF Spaces requires port 7860
EXPOSE 7860
 
# Start FastAPI on 8000 in background, Streamlit on 7860 in foreground
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true"]