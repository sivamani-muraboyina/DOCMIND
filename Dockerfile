FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF, pdfplumber, and faiss-cpu (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1-mesa-glx libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data so it doesn't fail at runtime
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

COPY . .

RUN mkdir -p /app/uploads /app/data && chmod -R 777 /app/uploads /app/data

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]