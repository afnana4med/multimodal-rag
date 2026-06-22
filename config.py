"""Central configuration. Providers are auto-detected from .env; with no keys
the system runs fully local (sentence-transformers + OCR + extractive)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
IMAGE_DIR = DATA_DIR / "images"
CHROMA_DIR = DATA_DIR / "chromadb"


def ensure_dirs() -> None:
    for d in (PDF_DIR, IMAGE_DIR, CHROMA_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --- API keys (optional; presence flips a stage to that provider) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip() or None
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip() or None
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip() or None  # OpenAI-compatible
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

HAS_OPENAI = OPENAI_API_KEY is not None
HAS_ANTHROPIC = ANTHROPIC_API_KEY is not None
HAS_GROQ = GROQ_API_KEY is not None

# --- Embeddings (same model for text AND image summaries — see README) ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai" if HAS_OPENAI else "local")
if EMBEDDING_PROVIDER == "openai":
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
else:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# --- Vision (image -> searchable text). Priority: anthropic > openai > groq > ocr ---
if HAS_ANTHROPIC:
    VISION_PROVIDER = os.getenv("VISION_PROVIDER", "anthropic")
elif HAS_OPENAI:
    VISION_PROVIDER = os.getenv("VISION_PROVIDER", "openai")
elif HAS_GROQ:
    VISION_PROVIDER = os.getenv("VISION_PROVIDER", "groq")
else:
    VISION_PROVIDER = os.getenv("VISION_PROVIDER", "ocr")

ANTHROPIC_VISION_MODEL = os.getenv("ANTHROPIC_VISION_MODEL", "claude-sonnet-4-6")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# --- Synthesis. Priority: anthropic > openai > groq > extractive ---
if HAS_ANTHROPIC:
    SYNTHESIS_PROVIDER = os.getenv("SYNTHESIS_PROVIDER", "anthropic")
elif HAS_OPENAI:
    SYNTHESIS_PROVIDER = os.getenv("SYNTHESIS_PROVIDER", "openai")
elif HAS_GROQ:
    SYNTHESIS_PROVIDER = os.getenv("SYNTHESIS_PROVIDER", "groq")
else:
    SYNTHESIS_PROVIDER = os.getenv("SYNTHESIS_PROVIDER", "extractive")

ANTHROPIC_SYNTHESIS_MODEL = os.getenv("ANTHROPIC_SYNTHESIS_MODEL", "claude-sonnet-4-6")
OPENAI_SYNTHESIS_MODEL = os.getenv("OPENAI_SYNTHESIS_MODEL", "gpt-4o")
GROQ_SYNTHESIS_MODEL = os.getenv("GROQ_SYNTHESIS_MODEL", "llama-3.3-70b-versatile")

# --- Parsing / chunking knobs ---
MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "100"))
MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "100"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "multimodal_rag")


def provider_for_key(api_key: str | None) -> str | None:
    """Detect the LLM provider from a user-supplied key prefix (bring-your-own-key)."""
    k = (api_key or "").strip()
    if k.startswith("gsk_"):
        return "groq"
    if k.startswith("sk-ant-"):
        return "anthropic"
    if k.startswith("sk-"):
        return "openai"
    return None


def summary() -> str:
    return "\n".join([
        "Multimodal RAG configuration",
        f"  embeddings : {EMBEDDING_PROVIDER:<10} ({EMBEDDING_MODEL})",
        f"  vision     : {VISION_PROVIDER}",
        f"  synthesis  : {SYNTHESIS_PROVIDER}",
        f"  openai key : {'yes' if HAS_OPENAI else 'no'}",
        f"  claude key : {'yes' if HAS_ANTHROPIC else 'no'}",
        f"  groq key   : {'yes' if HAS_GROQ else 'no'}",
    ])


if __name__ == "__main__":
    ensure_dirs()
    print(summary())
