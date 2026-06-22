"""Hybrid retrieval: vector search + BM25 keyword search fused with Reciprocal
Rank Fusion. Optionally scoped to a single document."""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rank_bm25 import BM25Okapi

import config
from indexing.embed import get_embedder
from indexing.store import get_collection

RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """Loads the collection into memory once and serves fused vector+keyword search."""

    def __init__(self):
        data = get_collection().get(include=["documents", "metadatas"])
        self.ids: list[str] = data["ids"]
        self.documents: list[str] = data["documents"]
        self.metadatas: list[dict] = data["metadatas"]
        self._by_id = {i: (d, m) for i, d, m in zip(self.ids, self.documents, self.metadatas)}
        self._bm25 = BM25Okapi([_tokenize(d) for d in self.documents]) if self.documents else None
        self.embedder = get_embedder()

    def _vector_ranking(self, query: str, n: int, doc_id: str | None = None) -> list[str]:
        where = {"doc_id": doc_id} if doc_id else None
        res = get_collection().query(
            query_embeddings=[self.embedder.embed_query(query)], n_results=n, where=where, include=[])
        return res["ids"][0]

    def _bm25_ranking(self, query: str, n: int, doc_id: str | None = None) -> list[str]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[str] = []
        for i in ranked:
            if doc_id and self.metadatas[i].get("doc_id") != doc_id:
                continue
            out.append(self.ids[i])
            if len(out) >= n:
                break
        return out

    @staticmethod
    def _rrf(rankings: list[list[str]]) -> dict[str, float]:
        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        return fused

    def search(self, query: str, k: int = 5, candidate_pool: int = 20,
               rerank: bool = False, doc_id: str | None = None) -> list[dict]:
        vec_ids = self._vector_ranking(query, candidate_pool, doc_id)
        bm25_ids = self._bm25_ranking(query, candidate_pool, doc_id)
        fused = self._rrf([vec_ids, bm25_ids])

        results = []
        for doc_id_, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
            doc, meta = self._by_id[doc_id_]
            results.append({
                "id": doc_id_, "document": doc, "metadata": meta,
                "type": meta.get("type"), "page": meta.get("page"),
                "image_path": meta.get("image_path") or None, "score": score,
            })

        if rerank:
            from retrieval.reranker import rerank_results
            results = rerank_results(query, results[: max(k, candidate_pool)])
        return results[:k]


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()


if __name__ == "__main__":
    r = HybridRetriever()
    for q in ["what does Figure 1 show?", "WMT 2014 English-to-German BLEU"]:
        print(f"\nQUERY: {q}")
        for hit in r.search(q, k=4):
            m = hit["metadata"]
            print(f'   {hit["score"]:.4f}  {m["type"]:5} p{m["page"]}  {hit["document"][:60].strip()!r}')
