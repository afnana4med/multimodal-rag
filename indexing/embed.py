"""
embed.py — turn text into vectors (brief §5, Day 3).

An embedding maps a piece of text to a point in high-dimensional space such
that semantically similar texts land near each other. Retrieval then becomes
"find the nearest points to the query."

THE critical rule (brief §5): text chunks AND image summaries must be embedded
with the SAME model. Otherwise their vectors live in different geometric spaces
and cosine similarity between them is meaningless. We guarantee this by sending
*everything* — query, text, image descriptions, tables — through one Embedder
instance chosen by config.

Two interchangeable backends:
  * LocalEmbedder  -> sentence-transformers, runs offline & free (default).
  * OpenAIEmbedder -> text-embedding-3-small (used when OPENAI_API_KEY is set).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


class Embedder:
    """Interface. `name` is stored in ChromaDB metadata so we never silently
    mix models across re-indexes (a future migration can detect a mismatch)."""

    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class LocalEmbedder(Embedder):
    """sentence-transformers. Model downloads once to ~/.cache, then offline."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)
        # bge models were trained with an instruction prefix on the QUERY side
        # only — adding it measurably improves retrieval. Harmless for others.
        self._query_prefix = (
            "Represent this sentence for searching relevant passages: "
            if "bge" in model_name.lower()
            else ""
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # normalize -> vectors are unit length, so dot product == cosine sim.
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode(
            self._query_prefix + text, normalize_embeddings=True, show_progress_bar=False
        )
        return vec.tolist()


class OpenAIEmbedder(Embedder):
    """OpenAI text-embedding-3-small (1536-dim). Needs OPENAI_API_KEY."""

    def __init__(self, model_name: str):
        from openai import OpenAI

        self.name = model_name
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.name, input=texts)
        return [d.embedding for d in resp.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return the configured embedder (built once and reused)."""
    if config.EMBEDDING_PROVIDER == "openai":
        return OpenAIEmbedder(config.EMBEDDING_MODEL)
    return LocalEmbedder(config.EMBEDDING_MODEL)


if __name__ == "__main__":
    emb = get_embedder()
    qv = emb.embed_query("multi-head attention mechanism")
    print(f"embedder: {emb.name}")
    print(f"vector dim: {len(qv)}")
    print(f"first 5 dims: {[round(x, 4) for x in qv[:5]]}")
