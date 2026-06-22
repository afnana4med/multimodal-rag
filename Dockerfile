# Hugging Face Spaces (Docker SDK) — runs the API + Streamlit UI in one container.
FROM python:3.11-slim

# System deps for PDF parsing + OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr poppler-utils curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Writable cache locations (HF Spaces containers need explicit, writable paths)
ENV HOME=/app \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data/pdfs data/images data/chromadb .cache && chmod -R 777 data .cache

# HF Spaces serves the app on port 7860
EXPOSE 7860
CMD ["bash", "start.sh"]
