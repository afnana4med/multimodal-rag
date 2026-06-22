"""FastAPI service: ingest PDFs and answer questions. The UI is a thin client
over these endpoints, so the two can be deployed separately."""

from __future__ import annotations

import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from indexing.store import get_collection, ingest_pdf, stats
from ingestion.summarize_images import summarize_document
from retrieval.retriever import get_retriever
from synthesis.answerer import ask

config.ensure_dirs()
app = FastAPI(title="Multimodal RAG", version="1.0")
app.mount("/images", StaticFiles(directory=str(config.IMAGE_DIR)), name="images")


class QueryRequest(BaseModel):
    query: str
    k: int = 5
    rerank: bool = False
    doc_id: str | None = None


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
    data = get_collection().get(include=["metadatas"])
    counts = Counter(m["doc_id"] for m in data["metadatas"])
    return {"documents": [{"doc_id": d, "units": n} for d, n in sorted(counts.items())]}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    dest = config.PDF_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    summarize_document(dest)
    n = ingest_pdf(dest)
    get_retriever.cache_clear()  # rebuild with the new document on next query
    return {"doc_id": dest.stem, "units_indexed": n, **stats()}


@app.post("/query")
def query(req: QueryRequest):
    res = ask(req.query, k=req.k, rerank=req.rerank, doc_id=req.doc_id)
    data = res.model_dump()
    data["image_files"] = [Path(p).name for p in res.image_evidence]
    return data
