"""
parse_pdf.py — turn a PDF into a structured list of multimodal Elements.

The whole premise of this project (brief §2): standard RAG flattens a PDF into
a text stream and loses the charts, tables, and diagrams — often 30–40% of the
information. So instead of "PDF -> text", we do "PDF -> [text | image | table]
elements", each tagged with the page it came from.

We use two lightweight libraries instead of a heavyweight ML layout model:
  * PyMuPDF (imported as `fitz`) -> page text + embedded images
  * pdfplumber                  -> tables (via the page's line geometry)

Run it:
    python ingestion/parse_pdf.py data/pdfs/attention_is_all_you_need.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, Optional

# Make `from config import ...` work when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz  # PyMuPDF
import pdfplumber
from pydantic import BaseModel

import config

ElementType = Literal["text", "image", "table"]


class Element(BaseModel):
    """One piece of a document. The atomic unit everything downstream uses.

    `content` holds searchable text:
      * text  -> the page/paragraph text itself
      * table -> a plain-text rendering of the table grid
      * image -> EMPTY for now; Day 2's vision step fills it with a description
    """

    doc_id: str                      # e.g. "attention_is_all_you_need"
    element_id: str                  # unique, e.g. "attention..._p3_image_0"
    type: ElementType
    page: int                        # 1-indexed page number (for citations)
    content: str = ""                # searchable text (see note above)
    image_path: Optional[str] = None  # set for type == "image"


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
def extract_text(doc: fitz.Document, doc_id: str) -> list[Element]:
    """One text Element per page (kept page-level so citations stay precise).

    We do NOT chunk here — chunking into ~512-token pieces happens later in
    chunk_text.py. Keeping whole-page text now preserves clean provenance.
    """
    elements: list[Element] = []
    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        text = page.get_text("text").strip()
        if not text:
            continue  # image-only page — nothing textual to keep
        elements.append(
            Element(
                doc_id=doc_id,
                element_id=f"{doc_id}_p{page_num}_text",
                type="text",
                page=page_num,
                content=text,
            )
        )
    return elements


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------
def extract_images(doc: fitz.Document, doc_id: str) -> list[Element]:
    """Save every (sufficiently large) embedded image to data/images/.

    PyMuPDF stores images by an internal cross-reference id ("xref"). The same
    image can be reused on many pages (a header logo, say), so we dedupe by
    xref and keep the first page it appears on for provenance.
    """
    elements: list[Element] = []
    seen_xrefs: set[int] = set()
    counter = 0

    for page_index in range(len(doc)):
        page_num = page_index + 1
        page = doc[page_index]
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                pix = fitz.Pixmap(doc, xref)
            except Exception:
                continue  # broken/unsupported image stream — skip

            # Filter out tiny graphics (logos, rules, bullet icons).
            if pix.width < config.MIN_IMAGE_WIDTH or pix.height < config.MIN_IMAGE_HEIGHT:
                continue

            # Normalize colorspace so PNG saving always works:
            # CMYK (n-alpha >= 4) must be converted to RGB first.
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            out_path = config.IMAGE_DIR / f"{doc_id}_p{page_num}_img{counter}.png"
            try:
                pix.save(out_path)
            except Exception:
                continue
            finally:
                pix = None  # free memory

            elements.append(
                Element(
                    doc_id=doc_id,
                    element_id=f"{doc_id}_p{page_num}_image_{counter}",
                    type="image",
                    page=page_num,
                    content="",  # filled in on Day 2 by the vision step
                    image_path=str(out_path),
                )
            )
            counter += 1

    return elements


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------
def _render_table(rows: list[list[Optional[str]]]) -> str:
    """Render a grid of cells as readable pipe-delimited text.

    This text is what gets embedded/searched. A later (optional) vision pass
    can produce a richer natural-language summary, but even this plain form is
    far better than losing the table entirely.
    """
    lines = []
    for row in rows:
        cells = [(c or "").replace("\n", " ").strip() for c in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def extract_tables(pdf_path: Path, doc_id: str) -> list[Element]:
    """Detect tables per page with pdfplumber and serialize each to text."""
    elements: list[Element] = []
    counter = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_num = page_index + 1
            for table in page.extract_tables():
                if not table or len(table) < 2:
                    continue
                # Reject figure/plot grids masquerading as tables. The attention
                # heatmaps in this paper detect as 27–59 *columns* wide — no real
                # data table is that wide. (Financial tables are typically <12.)
                n_cols = max(len(r) for r in table)
                if n_cols > 15:
                    continue
                # Require enough genuine content: cells with >=3 chars filter out
                # both empty grids and reversed/fragmented figure text.
                cells = [c for row in table for c in row]
                non_empty = sum(1 for c in cells if c and c.strip())
                meaningful = sum(1 for c in cells if c and len(c.strip()) >= 3)
                if meaningful < 4 or non_empty / len(cells) < 0.20:
                    continue
                rendered = _render_table(table)
                if not rendered.strip():
                    continue
                elements.append(
                    Element(
                        doc_id=doc_id,
                        element_id=f"{doc_id}_p{page_num}_table_{counter}",
                        type="table",
                        page=page_num,
                        content=rendered,
                    )
                )
                counter += 1
    return elements


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def parse_pdf(pdf_path: str | Path) -> list[Element]:
    """Parse a PDF into text + image + table Elements."""
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
    """Write parsed elements to data/<doc_id>_elements.json for reuse by later
    stages (so we don't re-parse the PDF every run)."""
    out = config.DATA_DIR / f"{doc_id}_elements.json"
    out.write_text(
        json.dumps([e.model_dump() for e in elements], indent=2, ensure_ascii=False)
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a PDF into multimodal elements.")
    parser.add_argument("pdf", help="Path to the PDF (e.g. data/pdfs/foo.pdf)")
    parser.add_argument("--no-save", action="store_true", help="Don't write the JSON manifest")
    args = parser.parse_args()

    elements = parse_pdf(args.pdf)
    doc_id = Path(args.pdf).stem

    # --- Summary stats ---
    by_type = {"text": 0, "image": 0, "table": 0}
    for e in elements:
        by_type[e.type] += 1
    print(f"\nParsed '{doc_id}': {len(elements)} elements "
          f"({by_type['text']} text, {by_type['image']} image, {by_type['table']} table)\n")

    # --- Peek at a few elements of each type ---
    for etype in ("text", "image", "table"):
        sample = next((e for e in elements if e.type == etype), None)
        if sample is None:
            print(f"[{etype}] none found")
            continue
        if etype == "image":
            print(f"[image] page {sample.page} -> {sample.image_path}")
        else:
            snippet = sample.content.replace("\n", " ")[:160]
            print(f"[{etype}] page {sample.page}: {snippet}...")
    print()

    if not args.no_save:
        path = save_manifest(elements, doc_id)
        print(f"Manifest written -> {path}")
        print(f"Images saved under -> {config.IMAGE_DIR}")


if __name__ == "__main__":
    main()
