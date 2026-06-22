# Hugging Face Spaces (Docker SDK) — runs the API + Streamlit UI in one container.
FROM python:3.11-slim

# System deps for PDF parsing + OCR (need root)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr poppler-utils curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces convention: run as non-root user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/sentence-transformers \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
RUN mkdir -p data/pdfs data/images data/chromadb

EXPOSE 7860
CMD ["bash", "start.sh"]
