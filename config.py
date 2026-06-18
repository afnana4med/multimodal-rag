"""
Central configuration for the multimodal RAG system.

Why this file exists
--------------------
Every other module imports from here instead of reading os.environ directly.
That gives us ONE place that decides:
  * where files live on disk (PDFs, extracted images, the vector DB), and
  * which provider/model to use for embeddings, vision, and synthesis.

The key idea: providers are *auto-detected* from your .env. If you have no
API keys, the system runs 100% free and local (sentence-transformers + OCR).
The moment you add OPENAI_API_KEY or ANTHROPIC_API_KEY, the relevant stage
"upgrades" to that provider — with zero code changes anywhere else.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env (if present) into os.environ. Safe to call even if .env is missing.
load_dotenv()


# ---------------------------------------------------------------------------
# 1. Filesystem paths — everything is relative to the project root, so the
#    code works no matter what directory you run it from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
IMAGE_DIR = DATA_DIR / "images"
CHROMA_DIR = DATA_DIR / "chromadb"


def ensure_dirs() -> None:
    """Create the data directories if they don't exist yet."""
    for d in (PDF_DIR, IMAGE_DIR, CHROMA_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. API keys (optional). Presence of a key is what flips a stage from the
#    free local path to the hosted-provider path.
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip() or None
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip() or None

HAS_OPENAI = OPENAI_API_KEY is not None
HAS_ANTHROPIC = ANTHROPIC_API_KEY is not None


# ---------------------------------------------------------------------------
# 3. Embeddings.
#    CRITICAL RULE (see brief §5): text chunks and image summaries MUST use
#    the SAME embedding model, or their vectors live in different geometric
#    spaces and similarity scores become meaningless. This single setting is
#    used for both — that's how we guarantee the rule holds.
# ---------------------------------------------------------------------------
# "openai" -> text-embedding-3-small (needs key);  "local" -> free, on-device.
EMBEDDING_PROVIDER = os.getenv(
    "EMBEDDING_PROVIDER", "openai" if HAS_OPENAI else "local"
)

if EMBEDDING_PROVIDER == "openai":
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
else:
    # bge-small is a strong, tiny (~130MB) free retrieval model. Downloads
    # once to ~/.cache on first use, then runs offline on CPU.
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


# ---------------------------------------------------------------------------
# 4. Vision (turn a chart/table/diagram image into searchable text).
#    Priority: Anthropic (Claude) > OpenAI (GPT-4o) > free OCR fallback.
# ---------------------------------------------------------------------------
if HAS_ANTHROPIC:
    VISION_PROVIDER = os.getenv("VISION_PROVIDER", "anthropic")
elif HAS_OPENAI:
    VISION_PROVIDER = os.getenv("VISION_PROVIDER", "openai")
else:
    VISION_PROVIDER = os.getenv("VISION_PROVIDER", "ocr")

# Default models per provider (overridable via .env).
ANTHROPIC_VISION_MODEL = os.getenv("ANTHROPIC_VISION_MODEL", "claude-sonnet-4-6")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")


# ---------------------------------------------------------------------------
# 5. Answer synthesis (the final "write the answer" LLM call).
#    Same priority order. With no key we fall back to "extractive": we return
#    the best retrieved passages directly instead of an LLM-written answer.
# ---------------------------------------------------------------------------
if HAS_ANTHROPIC:
    SYNTHESIS_PROVIDER = os.getenv("SYNTHESIS_PROVIDER", "anthropic")
elif HAS_OPENAI:
    SYNTHESIS_PROVIDER = os.getenv("SYNTHESIS_PROVIDER", "openai")
else:
    SYNTHESIS_PROVIDER = os.getenv("SYNTHESIS_PROVIDER", "extractive")

ANTHROPIC_SYNTHESIS_MODEL = os.getenv("ANTHROPIC_SYNTHESIS_MODEL", "claude-sonnet-4-6")
OPENAI_SYNTHESIS_MODEL = os.getenv("OPENAI_SYNTHESIS_MODEL", "gpt-4o")


# ---------------------------------------------------------------------------
# 6. Parsing / chunking knobs.
# ---------------------------------------------------------------------------
# Skip extracted images smaller than this (px) — filters out logos, rules,
# bullet icons, and other non-informative graphics.
MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "100"))
MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "100"))

# Text chunking (used on Day 3).
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))      # ~tokens
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ChromaDB collection name.
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "multimodal_rag")


def summary() -> str:
    """Human-readable snapshot of the active configuration (for logging)."""
    lines = [
        "Multimodal RAG configuration",
        f"  embeddings : {EMBEDDING_PROVIDER:<10} ({EMBEDDING_MODEL})",
        f"  vision     : {VISION_PROVIDER}",
        f"  synthesis  : {SYNTHESIS_PROVIDER}",
        f"  openai key : {'yes' if HAS_OPENAI else 'no'}",
        f"  claude key : {'yes' if HAS_ANTHROPIC else 'no'}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ensure_dirs()
    print(summary())
