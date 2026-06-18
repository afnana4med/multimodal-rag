"""
answerer.py — final answer synthesis with image evidence (brief §6, Day 5).

The last stage of RAG: take the retrieved evidence and write a grounded answer.
What makes THIS multimodal: when a top hit is an image (a chart/diagram), we
attach the actual image bytes to the LLM call — so the model reasons over the
real pixels, not just our text description of them. The answer can then cite
both page numbers and figure evidence.

Cost guardrail (brief §12): we only attach an image when it is BOTH type
"image" AND ranked in the top 3. Images are expensive in tokens; don't send
them blindly.

Provider (auto from config):
  anthropic  -> Claude (claude-sonnet-4-6), images attached
  openai     -> GPT-4o, images attached
  extractive -> no LLM: return the best retrieved passages (free fallback)

    from synthesis.answerer import ask
    ask("How does multi-head attention work?")
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel

import config
from retrieval.retriever import get_retriever

SYSTEM_PROMPT = """You answer questions about a document using ONLY the provided context (text passages and any attached images).

Rules:
- Ground every claim in the context. If the context does not contain the answer, say so plainly — do not invent facts.
- Cite the page number for each fact in square brackets, e.g. [p.8].
- If an attached image (chart/diagram/table) supports the answer, reference it explicitly (e.g. "the architecture diagram on p.3 shows ...").
- Be specific with numbers and labels. Prefer concrete values over vague descriptions.
- Keep the answer concise and directly responsive to the question."""


class AnswerResult(BaseModel):
    answer: str
    citations: list[int]          # page numbers referenced (sorted, unique)
    image_evidence: list[str]     # image file paths attached as evidence
    provider: str                 # which backend produced the answer
    contexts: list[dict]          # the retrieved hits used (for transparency/UI)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _format_contexts(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        blocks.append(f"[{i}] (page {m['page']}, {m['type']})\n{h['document']}")
    return "\n\n".join(blocks)


def _select_image_evidence(hits: list[dict], max_images: int = 3) -> list[str]:
    """Brief §12: attach an image only if it's type=image AND in the top 3."""
    paths = []
    for h in hits[:3]:
        if h["metadata"].get("type") == "image":
            p = h["metadata"].get("image_path") or ""
            if p and Path(p).exists():
                paths.append(p)
    return paths[:max_images]


def _citations(hits: list[dict]) -> list[int]:
    return sorted({int(h["metadata"]["page"]) for h in hits})


def _b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Provider backends
# ---------------------------------------------------------------------------
def _answer_anthropic(query: str, hits: list[dict], images: list[str]) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    content: list[dict] = [{"type": "text", "text": f"Context:\n{_format_contexts(hits)}"}]
    for p in images:
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": _b64(p)}})
    content.append({"type": "text", "text": f"Question: {query}"})

    resp = client.messages.create(
        model=config.ANTHROPIC_SYNTHESIS_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text.strip()


def _answer_openai(query: str, hits: list[dict], images: list[str]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    content: list[dict] = [{"type": "text", "text": f"Context:\n{_format_contexts(hits)}"}]
    for p in images:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_b64(p)}"}})
    content.append({"type": "text", "text": f"Question: {query}"})

    resp = client.chat.completions.create(
        model=config.OPENAI_SYNTHESIS_MODEL,
        max_tokens=1024,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": content}],
    )
    return resp.choices[0].message.content.strip()


def _answer_extractive(query: str, hits: list[dict], images: list[str]) -> str:
    """No-LLM fallback: present the strongest retrieved evidence with citations.
    Honest about what it is — this is retrieval output, not a written answer."""
    lines = ["(No synthesis LLM configured — showing the most relevant retrieved "
             "evidence. Add an OpenAI/Anthropic key in .env for a written answer.)\n"]
    for h in hits:
        m = h["metadata"]
        snippet = " ".join(h["document"].split())[:400]
        marker = f"  [image: {Path(m['image_path']).name}]" if m["type"] == "image" and m.get("image_path") else ""
        lines.append(f"• [p.{m['page']}] ({m['type']}){marker} {snippet}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def ask(query: str, k: int = 5, rerank: bool = False) -> AnswerResult:
    hits = get_retriever().search(query, k=k, rerank=rerank)
    images = _select_image_evidence(hits)
    provider = config.SYNTHESIS_PROVIDER

    if provider == "anthropic":
        text = _answer_anthropic(query, hits, images)
    elif provider == "openai":
        text = _answer_openai(query, hits, images)
    else:
        text = _answer_extractive(query, hits, images)

    return AnswerResult(
        answer=text,
        citations=_citations(hits),
        image_evidence=images,
        provider=provider,
        contexts=[{"page": h["metadata"]["page"], "type": h["metadata"]["type"],
                   "score": round(float(h["score"]), 4)} for h in hits],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against the indexed corpus.")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    res = ask(args.question, k=args.k, rerank=args.rerank)
    print(f"\n=== ANSWER (provider: {res.provider}) ===\n{res.answer}\n")
    print(f"Citations (pages): {res.citations}")
    if res.image_evidence:
        print("Image evidence:")
        for p in res.image_evidence:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
