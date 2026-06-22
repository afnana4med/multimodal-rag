"""ChromaDB vector store: one mixed-modality collection holding text, image
summaries, and tables, with the embedding model recorded on each record."""

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
    return _client().get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def add_elements(elements: list[Element], embedder: Embedder | None = None) -> int:
    embedder = embedder or get_embedder()
    indexable = [e for e in elements if e.content and e.content.strip()]
    if not indexable:
        return 0
    embeddings = embedder.embed_documents([e.content for e in indexable])
    get_collection().upsert(
        ids=[e.element_id for e in indexable],
        documents=[e.content for e in indexable],
        embeddings=embeddings,
        metadatas=[{
            "doc_id": e.doc_id, "page": e.page, "type": e.type,
            "image_path": e.image_path or "", "embedding_model": embedder.name,
        } for e in indexable],
    )
    return len(indexable)


def ingest_pdf(pdf_path: str | Path, embedder: Embedder | None = None) -> int:
    """Reuses the parse+summarize manifest if present; else parses fresh."""
    from ingestion.parse_pdf import parse_pdf

    pdf_path = Path(pdf_path)
    manifest = config.DATA_DIR / f"{pdf_path.stem}_elements.json"
    if manifest.exists():
        elements = [Element(**d) for d in json.loads(manifest.read_text())]
    else:
        elements = parse_pdf(pdf_path)
    n = add_elements(chunk_elements(elements), embedder)
    print(f"Ingested '{pdf_path.stem}': {n} units indexed "
          f"(collection now holds {get_collection().count()} total).")
    return n


def query(query_text: str, k: int = 5, where: dict | None = None,
          embedder: Embedder | None = None) -> list[dict]:
    embedder = embedder or get_embedder()
    res = get_collection().query(
        query_embeddings=[embedder.embed_query(query_text)], n_results=k, where=where,
        include=["documents", "metadatas", "distances"])
    return [{
        "id": res["ids"][0][i],
        "document": res["documents"][0][i],
        "metadata": res["metadatas"][0][i],
        "distance": res["distances"][0][i],
        "score": 1.0 - res["distances"][0][i],
    } for i in range(len(res["ids"][0]))]


def stats() -> dict:
    return {"collection": config.COLLECTION_NAME, "count": get_collection().count()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF into ChromaDB.")
    parser.add_argument("pdf")
    args = parser.parse_args()
    config.ensure_dirs()
    ingest_pdf(args.pdf)


if __name__ == "__main__":
    main()
