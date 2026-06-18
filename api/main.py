"""
api/main.py — FastAPI service exposing the RAG pipeline (brief §6, Day 6).

Endpoints:
  GET  /            health + active config
  GET  /stats       how many units are indexed
  POST /ingest      upload a PDF -> parse -> summarize -> chunk -> index
  POST /query       ask a question -> grounded answer + citations + evidence
  /images/*         static serving of extracted images (so a remote UI can
                    render image evidence by URL)

Design: this service owns ALL the logic. The Streamlit UI is a thin client
that only makes HTTP calls — so the two can be deployed and scaled separately.

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from collections import Counter

import config
from indexing.store import get_collection, ingest_pdf, stats
from ingestion.summarize_images import summarize_document
from retrieval.retriever import get_retriever
from synthesis.answerer import ask

config.ensure_dirs()
app = FastAPI(title="Multimodal RAG", version="1.0")

# Serve extracted images so a deployed UI can show image evidence via URL.
app.mount("/images", StaticFiles(directory=str(config.IMAGE_DIR)), name="images")


class QueryRequest(BaseModel):
    query: str
    k: int = 5
    rerank: bool = False
    doc_id: str | None = None   # scope the query to a single document


@app.get("/")
def root():
    return {
        "service": "multimodal-rag",
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "embedding_model": config.EMBEDDING_MODEL,
        "vision_provider": config.VISION_PROVIDER,
        "synthesis_provider": config.SYNTHESIS_PROVIDER,
    }


@app.get("/stats")
def get_stats():
    return stats()


@app.get("/documents")
def list_documents():
    """List the distinct documents in the index (for the UI's doc selector)."""
    data = get_collection().get(include=["metadatas"])
    counts = Counter(m["doc_id"] for m in data["metadatas"])
    return {"documents": [{"doc_id": d, "units": n} for d, n in sorted(counts.items())]}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF and run the full ingestion pipeline synchronously."""
    dest = config.PDF_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # parse + summarize images (writes the manifest), then chunk + index.
    summarize_document(dest)
    n = ingest_pdf(dest)
    # The retriever caches the whole corpus + BM25 index; drop it so the next
    # query rebuilds with the newly added document included.
    get_retriever.cache_clear()
    return {"doc_id": dest.stem, "units_indexed": n, **stats()}


@app.post("/query")
def query(req: QueryRequest):
    res = ask(req.query, k=req.k, rerank=req.rerank, doc_id=req.doc_id)
    data = res.model_dump()
    # Expose just the basename so the UI can build /images/<name> URLs.
    data["image_files"] = [Path(p).name for p in res.image_evidence]
    return data
