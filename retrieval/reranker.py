"""
reranker.py — optional cross-encoder reranking (brief §6, Day 4, optional).

A bi-encoder (our embedding model) encodes the query and each document
*separately*, then compares vectors — fast, but it never sees the two together.
A cross-encoder feeds "query [SEP] document" through one transformer and outputs
a single relevance score. It's far more accurate but far slower, so the standard
pattern is: cheap retriever proposes ~20 candidates, the cross-encoder re-scores
just those and reorders the top.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (small, CPU-friendly; ~80MB on
first use). Entirely optional — the system works without it.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)


def rerank_results(query: str, results: list[dict]) -> list[dict]:
    """Re-score candidate results with the cross-encoder and sort by it.
    Adds a 'rerank_score' field; leaves the original fused 'score' intact."""
    if not results:
        return results
    pairs = [(query, r["document"]) for r in results]
    scores = _model().predict(pairs)
    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)
    return sorted(results, key=lambda r: r["rerank_score"], reverse=True)


if __name__ == "__main__":
    from retrieval.retriever import HybridRetriever

    r = HybridRetriever()
    q = "how does multi-head attention work?"
    print(f"QUERY: {q}\n")
    hits = r.search(q, k=5, rerank=True)
    for h in hits:
        print(f'   rerank={h.get("rerank_score", 0):.3f}  fused={h["score"]:.4f}  '
              f'{h["metadata"]["type"]:5} p{h["page"]}  {h["document"][:55].strip()!r}')
