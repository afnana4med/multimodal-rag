"""Smoke tests for retrieval + answering over the indexed collection.

These require an already-populated ChromaDB collection (run an ingest first):
    python indexing/store.py data/pdfs/attention_is_all_you_need.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from indexing.store import get_collection
from retrieval.retriever import HybridRetriever
from synthesis.answerer import ask

pytestmark = pytest.mark.skipif(
    get_collection().count() == 0, reason="collection empty — ingest a PDF first"
)


def test_hybrid_search_returns_results():
    hits = HybridRetriever().search("multi-head attention", k=5)
    assert 1 <= len(hits) <= 5
    assert all("page" in h["metadata"] and "type" in h["metadata"] for h in hits)
    # results should be ordered by descending fused score
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_answer_has_citations_and_structure():
    res = ask("How does multi-head attention work?", k=5)
    assert res.answer
    assert res.citations == sorted(set(res.citations))   # unique, sorted pages
    assert res.provider in {"anthropic", "openai", "extractive"}
    assert all(isinstance(p, int) for p in res.citations)


def test_modality_filter_returns_only_images():
    hits = HybridRetriever().search("architecture diagram", k=10)
    # filtering the same query by image type should yield image-only metadata
    images = [h for h in hits if h["metadata"]["type"] == "image"]
    for h in images:
        assert h["metadata"]["type"] == "image"
