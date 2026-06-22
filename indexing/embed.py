"""Text -> vector embeddings. The same Embedder is used for queries, text chunks,
and image summaries so everything lives in one comparable vector space."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


class Embedder:
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class LocalEmbedder(Embedder):
    """sentence-transformers; runs offline after first download."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)
        # bge models expect an instruction prefix on the query side only.
        self._query_prefix = (
            "Represent this sentence for searching relevant passages: "
            if "bge" in model_name.lower() else ""
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(self._query_prefix + text, normalize_embeddings=True,
                                 show_progress_bar=False).tolist()


class OpenAIEmbedder(Embedder):
    def __init__(self, model_name: str):
        from openai import OpenAI

        self.name = model_name
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [d.embedding for d in self.client.embeddings.create(model=self.name, input=texts).data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if config.EMBEDDING_PROVIDER == "openai":
        return OpenAIEmbedder(config.EMBEDDING_MODEL)
    return LocalEmbedder(config.EMBEDDING_MODEL)


if __name__ == "__main__":
    emb = get_embedder()
    qv = emb.embed_query("multi-head attention mechanism")
    print(f"embedder: {emb.name}  dim: {len(qv)}")
