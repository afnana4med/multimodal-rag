"""Optional cross-encoder reranker. Re-scores the candidate pool by reading
query+passage together — slower but more precise than the bi-encoder."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)


def rerank_results(query: str, results: list[dict]) -> list[dict]:
    if not results:
        return results
    scores = _model().predict([(query, r["document"]) for r in results])
    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)
    return sorted(results, key=lambda r: r["rerank_score"], reverse=True)


if __name__ == "__main__":
    from retrieval.retriever import HybridRetriever

    q = "how does multi-head attention work?"
    print(f"QUERY: {q}\n")
    for h in HybridRetriever().search(q, k=5, rerank=True):
        print(f'   rerank={h.get("rerank_score", 0):.3f}  {h["metadata"]["type"]:5} '
              f'p{h["page"]}  {h["document"][:55].strip()!r}')
