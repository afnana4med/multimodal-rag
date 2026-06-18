"""Smoke tests for the ingestion stage (parsing -> elements)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from ingestion.chunk_text import chunk_elements
from ingestion.parse_pdf import Element, parse_pdf

TEST_PDF = Path(__file__).resolve().parent.parent / "data" / "pdfs" / "attention_is_all_you_need.pdf"
pytestmark = pytest.mark.skipif(not TEST_PDF.exists(), reason="test PDF not present")


def test_parse_returns_multimodal_elements():
    els = parse_pdf(TEST_PDF)
    assert els, "expected non-empty element list"
    types = {e.type for e in els}
    assert "text" in types                      # every real doc has text
    assert types <= {"text", "image", "table"}  # no unexpected element types
    assert all(e.page >= 1 for e in els)         # 1-indexed pages for citations
    assert all(isinstance(e, Element) for e in els)


def test_images_have_paths_and_exist():
    images = [e for e in parse_pdf(TEST_PDF) if e.type == "image"]
    for img in images:
        assert img.image_path
        assert Path(img.image_path).exists()


def test_chunking_preserves_pages_and_splits_only_text():
    els = parse_pdf(TEST_PDF)
    chunks = chunk_elements(els)
    # non-text elements pass through unchanged in count
    assert sum(1 for e in chunks if e.type != "text") == sum(1 for e in els if e.type != "text")
    # text is split into >= as many chunks as pages
    assert sum(1 for e in chunks if e.type == "text") >= sum(1 for e in els if e.type == "text")
    assert all(c.page >= 1 for c in chunks)
