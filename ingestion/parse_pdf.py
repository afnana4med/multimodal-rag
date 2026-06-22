"""Parse a PDF into structured elements (text / image / table), each tagged with
its page number. Uses PyMuPDF for text + images and pdfplumber for tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz  # PyMuPDF
import pdfplumber
from pydantic import BaseModel

import config

ElementType = Literal["text", "image", "table"]


class Element(BaseModel):
    """One indexable unit. `content` holds searchable text; for images it is
    empty until the vision step (summarize_images.py) fills it in."""

    doc_id: str
    element_id: str
    type: ElementType
    page: int
    content: str = ""
    image_path: Optional[str] = None


def extract_text(doc: fitz.Document, doc_id: str) -> list[Element]:
    """One text Element per page (chunking happens later in chunk_text.py)."""
    elements: list[Element] = []
    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        text = page.get_text("text").strip()
        if not text:
            continue
        elements.append(Element(
            doc_id=doc_id, element_id=f"{doc_id}_p{page_num}_text",
            type="text", page=page_num, content=text,
        ))
    return elements


def extract_images(doc: fitz.Document, doc_id: str) -> list[Element]:
    """Save each embedded image (above a size threshold) to data/images/.
    Images are deduped by xref; the first page they appear on is recorded."""
    elements: list[Element] = []
    seen_xrefs: set[int] = set()
    counter = 0

    for page_index in range(len(doc)):
        page_num = page_index + 1
        for img in doc[page_index].get_images(full=True):
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                pix = fitz.Pixmap(doc, xref)
            except Exception:
                continue
            if pix.width < config.MIN_IMAGE_WIDTH or pix.height < config.MIN_IMAGE_HEIGHT:
                continue
            if pix.n - pix.alpha >= 4:  # CMYK -> RGB so PNG saving works
                pix = fitz.Pixmap(fitz.csRGB, pix)
            out_path = config.IMAGE_DIR / f"{doc_id}_p{page_num}_img{counter}.png"
            try:
                pix.save(out_path)
            except Exception:
                continue
            finally:
                pix = None
            elements.append(Element(
                doc_id=doc_id, element_id=f"{doc_id}_p{page_num}_image_{counter}",
                type="image", page=page_num, content="", image_path=str(out_path),
            ))
            counter += 1
    return elements


def _render_table(rows: list[list[Optional[str]]]) -> str:
    lines = []
    for row in rows:
        cells = [(c or "").replace("\n", " ").strip() for c in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def extract_tables(pdf_path: Path, doc_id: str) -> list[Element]:
    """Detect tables with pdfplumber and serialize each to text. Filters out
    figure-grid false positives (very wide grids or near-empty cells)."""
    elements: list[Element] = []
    counter = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_num = page_index + 1
            for table in page.extract_tables():
                if not table or len(table) < 2:
                    continue
                if max(len(r) for r in table) > 15:  # plots/diagrams, not data tables
                    continue
                cells = [c for row in table for c in row]
                non_empty = sum(1 for c in cells if c and c.strip())
                meaningful = sum(1 for c in cells if c and len(c.strip()) >= 3)
                if meaningful < 4 or non_empty / len(cells) < 0.20:
                    continue
                rendered = _render_table(table)
                if not rendered.strip():
                    continue
                elements.append(Element(
                    doc_id=doc_id, element_id=f"{doc_id}_p{page_num}_table_{counter}",
                    type="table", page=page_num, content=rendered,
                ))
                counter += 1
    return elements


def parse_pdf(pdf_path: str | Path) -> list[Element]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    config.ensure_dirs()
    doc_id = pdf_path.stem

    doc = fitz.open(pdf_path)
    try:
        text_elements = extract_text(doc, doc_id)
        image_elements = extract_images(doc, doc_id)
    finally:
        doc.close()
    table_elements = extract_tables(pdf_path, doc_id)
    return text_elements + image_elements + table_elements


def save_manifest(elements: list[Element], doc_id: str) -> Path:
    out = config.DATA_DIR / f"{doc_id}_elements.json"
    out.write_text(json.dumps([e.model_dump() for e in elements], indent=2, ensure_ascii=False))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a PDF into multimodal elements.")
    parser.add_argument("pdf")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    elements = parse_pdf(args.pdf)
    doc_id = Path(args.pdf).stem
    by_type = {"text": 0, "image": 0, "table": 0}
    for e in elements:
        by_type[e.type] += 1
    print(f"\nParsed '{doc_id}': {len(elements)} elements "
          f"({by_type['text']} text, {by_type['image']} image, {by_type['table']} table)\n")
    for etype in ("text", "image", "table"):
        sample = next((e for e in elements if e.type == etype), None)
        if sample is None:
            print(f"[{etype}] none found")
        elif etype == "image":
            print(f"[image] page {sample.page} -> {sample.image_path}")
        else:
            print(f"[{etype}] page {sample.page}: {sample.content.replace(chr(10), ' ')[:160]}...")
    if not args.no_save:
        print(f"\nManifest -> {save_manifest(elements, doc_id)}")


if __name__ == "__main__":
    main()
