"""Turn extracted images into searchable text via a vision model (or OCR).
Provider auto-selected by config: anthropic > openai > groq > ocr."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from ingestion.parse_pdf import Element, parse_pdf, save_manifest

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


def _b64(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def _summarize_anthropic(image_path: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=config.ANTHROPIC_VISION_MODEL, max_tokens=1024,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(image_path)}},
            {"type": "text", "text": IMAGE_SUMMARY_PROMPT},
        ]}],
    )
    return resp.content[0].text.strip()


def _summarize_openai_compatible(image_path: str, api_key: str, base_url: str | None, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model, max_tokens=1024,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": IMAGE_SUMMARY_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(image_path)}"}},
        ]}],
    )
    return resp.choices[0].message.content.strip()


def _summarize_ocr(image_path: str) -> str:
    import pytesseract
    from PIL import Image

    text = " ".join(pytesseract.image_to_string(Image.open(image_path)).split())
    return f"[OCR-extracted text from image] {text}" if text else "[OCR found no readable text in this image]"


def summarize_image(image_path: str, provider: str | None = None) -> str:
    provider = provider or config.VISION_PROVIDER
    if provider == "anthropic":
        return _summarize_anthropic(image_path)
    if provider == "openai":
        return _summarize_openai_compatible(image_path, config.OPENAI_API_KEY, None, config.OPENAI_VISION_MODEL)
    if provider == "groq":
        return _summarize_openai_compatible(image_path, config.GROQ_API_KEY, config.GROQ_BASE_URL, config.GROQ_VISION_MODEL)
    if provider == "ocr":
        return _summarize_ocr(image_path)
    raise ValueError(f"Unknown vision provider: {provider}")


def _load_or_parse(pdf_path: Path) -> list[Element]:
    manifest = config.DATA_DIR / f"{pdf_path.stem}_elements.json"
    if manifest.exists():
        return [Element(**d) for d in json.loads(manifest.read_text())]
    return parse_pdf(pdf_path)


def summarize_document(pdf_path: str | Path, provider: str | None = None) -> list[Element]:
    """Fill in each image Element's text, then persist the manifest + a
    path->summary JSON."""
    pdf_path = Path(pdf_path)
    elements = _load_or_parse(pdf_path)
    provider = provider or config.VISION_PROVIDER
    image_elements = [e for e in elements if e.type == "image"]
    print(f"Summarizing {len(image_elements)} images with provider='{provider}'...")

    summaries: dict[str, str] = {}
    for i, el in enumerate(image_elements, 1):
        try:
            desc = summarize_image(el.image_path, provider)
        except Exception as exc:
            desc = f"[summary failed: {exc}]"
        el.content = desc
        summaries[el.image_path] = desc
        print(f"  [{i}/{len(image_elements)}] {Path(el.image_path).name}: {desc[:80]}...")

    save_manifest(elements, pdf_path.stem)
    out = config.DATA_DIR / f"{pdf_path.stem}_summaries.json"
    out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"\nSummaries -> {out}")
    return elements


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a PDF's extracted images into text.")
    parser.add_argument("pdf")
    parser.add_argument("--provider", choices=["anthropic", "openai", "groq", "ocr"])
    args = parser.parse_args()
    config.ensure_dirs()
    summarize_document(args.pdf, args.provider)


if __name__ == "__main__":
    main()
