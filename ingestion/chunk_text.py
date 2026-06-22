"""Split text Elements into token-sized chunks (with overlap); image/table
Elements pass through unchanged."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from ingestion.parse_pdf import Element


def _splitter() -> RecursiveCharacterTextSplitter:
    # chunk_size/overlap counted in tokens (tiktoken), matching embedding models.
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
    )


def chunk_elements(elements: list[Element]) -> list[Element]:
    splitter = _splitter()
    out: list[Element] = []
    for el in elements:
        if el.type != "text":
            out.append(el)
            continue
        for i, piece in enumerate(splitter.split_text(el.content)):
            out.append(Element(
                doc_id=el.doc_id, element_id=f"{el.element_id}_c{i}",
                type="text", page=el.page, content=piece,  # chunks inherit the page
            ))
    return out


if __name__ == "__main__":
    import json

    doc_id = "attention_is_all_you_need"
    elements = [Element(**d) for d in json.loads(
        (config.DATA_DIR / f"{doc_id}_elements.json").read_text())]
    chunks = chunk_elements(elements)
    print(f"{sum(e.type == 'text' for e in elements)} text pages -> "
          f"{sum(e.type == 'text' for e in chunks)} chunks; total units: {len(chunks)}")
