"""
chunk_text.py — split page text into embedding-sized chunks (brief §8, Day 3).

Why chunk at all? Two reasons:
  1. Embedding models have a token limit; a whole page can overflow it.
  2. Retrieval is more precise on focused passages than on giant blobs — a
     query about one sentence shouldn't have to match an entire page.

We split with RecursiveCharacterTextSplitter, which tries to break on natural
boundaries (paragraphs -> lines -> sentences -> words) so chunks stay coherent.
We measure size in *tokens* (via tiktoken) to match how embedding models count,
and we overlap consecutive chunks slightly so a fact straddling a boundary
still lands wholly inside at least one chunk.

Image and table Elements are NOT split — each is already one atomic unit
(an image description / a serialized table), so it passes straight through.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from ingestion.parse_pdf import Element


def _splitter() -> RecursiveCharacterTextSplitter:
    # from_tiktoken_encoder => chunk_size/overlap are counted in TOKENS, not chars.
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )


def chunk_elements(elements: list[Element]) -> list[Element]:
    """Split text Elements into chunk Elements; pass image/table through."""
    splitter = _splitter()
    out: list[Element] = []
    for el in elements:
        if el.type != "text":
            out.append(el)
            continue
        pieces = splitter.split_text(el.content)
        if not pieces:
            continue
        for i, piece in enumerate(pieces):
            out.append(
                Element(
                    doc_id=el.doc_id,
                    element_id=f"{el.element_id}_c{i}",
                    type="text",
                    page=el.page,           # chunks inherit their page -> citations stay correct
                    content=piece,
                )
            )
    return out


if __name__ == "__main__":
    import json

    # Quick demo on the test doc's manifest.
    doc_id = "attention_is_all_you_need"
    manifest = config.DATA_DIR / f"{doc_id}_elements.json"
    elements = [Element(**d) for d in json.loads(manifest.read_text())]
    chunks = chunk_elements(elements)
    n_text_before = sum(1 for e in elements if e.type == "text")
    n_text_after = sum(1 for e in chunks if e.type == "text")
    print(f"{n_text_before} text pages -> {n_text_after} text chunks "
          f"(+ {sum(1 for e in chunks if e.type != 'text')} image/table units)")
    print(f"total indexable units: {len(chunks)}")
