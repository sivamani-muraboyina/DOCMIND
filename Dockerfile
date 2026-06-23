FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF, pdfplumber, faiss-cpu (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data at build time so it never fails at runtime
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

COPY . .

# Create required dirs with open permissions
RUN mkdir -p /app/uploads /app/data/raw/corpus/cuad /app/data/legal_index \
    && chmod -R 777 /app/uploads /app/data

EXPOSE 10000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port ${PORT:-10000} --server.address 0.0.0.0 --server.headless true"]