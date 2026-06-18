"""
summarize_images.py — turn extracted images into searchable text (brief §2, §10).

This is the heart of "multimodal" RAG. An image embedding model (like CLIP)
is good at "find pictures that LOOK like X" but terrible at "find the picture
that ANSWERS this question" (brief §12). So instead we ask a vision-capable LLM
to *describe* each image in rich detail, then embed that description as text.
That description becomes the only way the image is found during search.

Provider is chosen automatically by config (no code changes to switch):
    anthropic -> Claude (claude-sonnet-4-6)   [best]
    openai    -> GPT-4o
    ocr       -> tesseract (FREE fallback; extracts on-image text only)

Run it (uses the manifest from parse_pdf.py, or parses the PDF if needed):
    python ingestion/summarize_images.py data/pdfs/attention_is_all_you_need.pdf
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from ingestion.parse_pdf import Element, parse_pdf, save_manifest

# The prompt is the single most important lever on retrieval quality (brief §10).
# It tells the vision model to be exhaustive and number-specific, because the
# description is the ONLY thing that gets indexed.
IMAGE_SUMMARY_PROMPT = """Describe this image in detail for retrieval purposes. Be comprehensive — this description will be the only way someone finds this image when searching.

If it's a CHART:
- Chart type (bar, line, pie, scatter, area, etc.)
- All axis labels with units
- Time period covered (e.g. "Q1 2020 through Q4 2024")
- Key data points and their values
- Trends, peaks, troughs, inflection points
- Comparisons being made between series
- Title, legend, and any caption text

If it's a TABLE:
- Headers (row and column)
- Number of rows
- Key data — transcribe the most important rows fully
- Units (currency, percentages, etc.)
- Footnotes if visible

If it's a DIAGRAM or FLOWCHART:
- What system or process it represents
- Components and how they connect
- Direction of flow
- Labels on arrows and nodes

If it's an INFOGRAPHIC or other:
- Main message it communicates
- Key statistics or quotes shown
- Source citations if visible

Be specific with numbers. Don't say "shows growth" — say "shows revenue grew from $50B to $89B between 2020 and 2024".

Return ONLY the description. No preamble."""


# ---------------------------------------------------------------------------
# Provider backends
# ---------------------------------------------------------------------------
def _b64(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def _summarize_anthropic(image_path: str) -> str:
    """Claude vision. Imported lazily so the library is only needed if used."""
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=config.ANTHROPIC_VISION_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": _b64(image_path),
                }},
                {"type": "text", "text": IMAGE_SUMMARY_PROMPT},
            ],
        }],
    )
    return resp.content[0].text.strip()


def _summarize_openai(image_path: str) -> str:
    """GPT-4o vision."""
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=config.OPENAI_VISION_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": IMAGE_SUMMARY_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{_b64(image_path)}",
                }},
            ],
        }],
    )
    return resp.choices[0].message.content.strip()


def _summarize_ocr(image_path: str) -> str:
    """FREE fallback: pull any text printed on the image via tesseract OCR.

    This is much weaker than a vision LLM — it reads box labels and axis text
    but can't describe trends or chart *meaning*. It keeps the pipeline fully
    runnable for $0, and is clearly marked so we know it's a placeholder.
    """
    import pytesseract
    from PIL import Image

    text = pytesseract.image_to_string(Image.open(image_path)).strip()
    # Collapse the noisy whitespace OCR tends to produce.
    text = " ".join(text.split())
    if not text:
        return "[OCR found no readable text in this image]"
    return f"[OCR-extracted text from image] {text}"


def summarize_image(image_path: str, provider: str | None = None) -> str:
    """Dispatch to the configured (or overridden) vision backend."""
    provider = provider or config.VISION_PROVIDER
    if provider == "anthropic":
        return _summarize_anthropic(image_path)
    if provider == "openai":
        return _summarize_openai(image_path)
    if provider == "ocr":
        return _summarize_ocr(image_path)
    raise ValueError(f"Unknown vision provider: {provider}")


# ---------------------------------------------------------------------------
# Batch over a document
# ---------------------------------------------------------------------------
def _load_or_parse(pdf_path: Path) -> list[Element]:
    """Reuse the parse_pdf manifest if it exists; otherwise parse fresh."""
    manifest = config.DATA_DIR / f"{pdf_path.stem}_elements.json"
    if manifest.exists():
        return [Element(**d) for d in json.loads(manifest.read_text())]
    return parse_pdf(pdf_path)


def summarize_document(pdf_path: str | Path, provider: str | None = None) -> list[Element]:
    """Fill in the `content` of every image Element with a text description,
    then persist both the updated manifest and a path->summary JSON."""
    pdf_path = Path(pdf_path)
    elements = _load_or_parse(pdf_path)
    provider = provider or config.VISION_PROVIDER

    image_elements = [e for e in elements if e.type == "image"]
    print(f"Summarizing {len(image_elements)} images with provider='{provider}'...")

    summaries: dict[str, str] = {}
    for i, el in enumerate(image_elements, 1):
        try:
            desc = summarize_image(el.image_path, provider)
        except Exception as exc:  # one bad image shouldn't kill the batch
            desc = f"[summary failed: {exc}]"
        el.content = desc
        summaries[el.image_path] = desc
        print(f"  [{i}/{len(image_elements)}] {Path(el.image_path).name}: {desc[:80]}...")

    # Persist: (1) updated manifest so Day 3 embeds image descriptions,
    #          (2) standalone summaries file = the Day 2 deliverable.
    save_manifest(elements, pdf_path.stem)
    out = config.DATA_DIR / f"{pdf_path.stem}_summaries.json"
    out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"\nSummaries -> {out}")
    return elements


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a PDF's extracted images into text.")
    parser.add_argument("pdf", help="Path to the PDF")
    parser.add_argument("--provider", choices=["anthropic", "openai", "ocr"],
                        help="Override the auto-detected vision provider")
    args = parser.parse_args()
    config.ensure_dirs()
    summarize_document(args.pdf, args.provider)


if __name__ == "__main__":
    main()
