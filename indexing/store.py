"""
store.py — the ChromaDB vector store wrapper (brief §6, §8 Day 3).

ChromaDB persists, for every indexable unit:
  * the text       (chunk text / image description / serialized table)
  * the embedding  (its vector, computed by embed.py)
  * metadata       (doc_id, page, modality, image path, embedding model)

Design choices that matter:
  * ONE collection holds all modalities (text + image + table). That's what
    lets a single query retrieve a paragraph AND a chart together (brief §6).
  * Cosine distance ("hnsw:space": "cosine"), matching our normalized vectors.
  * We store the embedding model name on every record (brief §5) so a future
    re-index can refuse to mix incompatible vector spaces.
  * upsert (not add) => re-ingesting a document updates rather than duplicates.

CLI — full end-to-end ingest of one PDF (parse -> summarize -> chunk -> store):
    python indexing/store.py data/pdfs/attention_is_all_you_need.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb

import config
from indexing.embed import Embedder, get_embedder
from ingestion.chunk_text import chunk_elements
from ingestion.parse_pdf import Element


@lru_cache(maxsize=1)
def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection():
    """Return (creating if needed) the single mixed-modality collection."""
    return _client().get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_elements(elements: list[Element], embedder: Embedder | None = None) -> int:
    """Embed and upsert elements that carry text content. Returns count added."""
    embedder = embedder or get_embedder()
    # Skip anything with no searchable text (e.g. an image OCR'd as empty).
    indexable = [e for e in elements if e.content and e.content.strip()]
    if not indexable:
        return 0

    embeddings = embedder.embed_documents([e.content for e in indexable])

    get_collection().upsert(
        ids=[e.element_id for e in indexable],
        documents=[e.content for e in indexable],
        embeddings=embeddings,
        metadatas=[
            {
                "doc_id": e.doc_id,
                "page": e.page,
                "type": e.type,                       # "text" | "image" | "table"
                "image_path": e.image_path or "",      # Chroma forbids None
                "embedding_model": embedder.name,      # guard against model drift
            }
            for e in indexable
        ],
    )
    return len(indexable)


def ingest_pdf(pdf_path: str | Path, embedder: Embedder | None = None) -> int:
    """Full pipeline for one PDF. Reuses the parse+summarize manifest if present
    (so images keep their vision/OCR descriptions); otherwise parses fresh."""
    from ingestion.parse_pdf import parse_pdf

    pdf_path = Path(pdf_path)
    manifest = config.DATA_DIR / f"{pdf_path.stem}_elements.json"
    if manifest.exists():
        elements = [Element(**d) for d in json.loads(manifest.read_text())]
    else:
        elements = parse_pdf(pdf_path)

    chunks = chunk_elements(elements)
    n = add_elements(chunks, embedder)
    print(f"Ingested '{pdf_path.stem}': {n} units indexed "
          f"(collection now holds {get_collection().count()} total).")
    return n


def query(query_text: str, k: int = 5, where: dict | None = None,
          embedder: Embedder | None = None) -> list[dict]:
    """Vector search. Returns k nearest units as dicts with score + metadata."""
    embedder = embedder or get_embedder()
    qvec = embedder.embed_query(query_text)
    res = get_collection().query(
        query_embeddings=[qvec],
        n_results=k,
        where=where,                 # e.g. {"type": "image"} to filter modality
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for i in range(len(res["ids"][0])):
        out.append({
            "id": res["ids"][0][i],
            "document": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],          # lower = closer
            "score": 1.0 - res["distances"][0][i],       # cosine similarity
        })
    return out


def stats() -> dict:
    col = get_collection()
    return {"collection": config.COLLECTION_NAME, "count": col.count()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF into ChromaDB.")
    parser.add_argument("pdf", help="Path to the PDF to ingest")
    args = parser.parse_args()
    config.ensure_dirs()
    ingest_pdf(args.pdf)


if __name__ == "__main__":
    main()
