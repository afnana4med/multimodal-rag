"""Answer synthesis. Builds a grounded answer from retrieved evidence and, when
useful, a chart spec whose numbers are extracted from the document context.
Provider auto-selected by config: anthropic > openai > groq > extractive."""

from __future__ import annotations

import argparse
import base64
import json
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
- If an attached image (chart/diagram/table) supports the answer, reference it explicitly.
- Be specific with numbers and labels. Prefer concrete values over vague descriptions.
- Keep the answer concise and directly responsive to the question."""

CHART_SYSTEM_PROMPT = SYSTEM_PROMPT + """

Additionally, return your response as a JSON object with exactly two keys:
- "answer": your markdown answer (with [p.X] citations), following the rules above.
- "chart": null, OR a chart spec visualizing data taken DIRECTLY from the context.
  Include a chart only if the user asks to plot/chart/graph/visualize/compare/show
  a breakdown or trend, OR when a numeric comparison clearly benefits from one.
  NEVER invent or estimate numbers — use exact values found in the context.
  Chart spec shape:
    {"type": "bar" | "line" | "pie",
     "title": "...", "x_label": "...", "y_label": "... (include units)",
     "categories": ["..."],
     "series": [{"name": "...", "values": [n, ...]}]}
  Values align 1:1 with categories. For pie, use a single series. Output ONLY the JSON object."""


class AnswerResult(BaseModel):
    answer: str
    citations: list[int]
    image_evidence: list[str]
    provider: str
    contexts: list[dict]
    chart: dict | None = None


def _validate_chart(chart) -> dict | None:
    """Defensively validate a model-produced chart spec; drop it if malformed."""
    if not isinstance(chart, dict) or chart.get("type") not in {"bar", "line", "pie"}:
        return None
    cats, series = chart.get("categories"), chart.get("series")
    if not isinstance(cats, list) or not cats or not isinstance(series, list) or not series:
        return None
    clean_series = []
    for s in series:
        vals = s.get("values") if isinstance(s, dict) else None
        if not isinstance(vals, list) or len(vals) != len(cats):
            continue
        try:
            vals = [float(v) for v in vals]
        except (TypeError, ValueError):
            continue
        clean_series.append({"name": str(s.get("name", "")), "values": vals})
    if not clean_series:
        return None
    return {
        "type": chart["type"], "title": str(chart.get("title", "")),
        "x_label": str(chart.get("x_label", "")), "y_label": str(chart.get("y_label", "")),
        "categories": [str(c) for c in cats], "series": clean_series,
    }


def _format_contexts(hits: list[dict], max_chars: int = 900) -> str:
    # Cap each chunk so large k values don't blow the model's token budget.
    blocks = []
    for i, h in enumerate(hits, 1):
        doc = h["document"]
        if len(doc) > max_chars:
            doc = doc[:max_chars] + "…"
        blocks.append(f"[{i}] (page {h['metadata']['page']}, {h['metadata']['type']})\n{doc}")
    return "\n\n".join(blocks)


def _evidence_lines(hits: list[dict]) -> str:
    out = []
    for h in hits:
        m = h["metadata"]
        snippet = " ".join(h["document"].split())[:400]
        marker = f"  [image: {Path(m['image_path']).name}]" if m["type"] == "image" and m.get("image_path") else ""
        out.append(f"• [p.{m['page']}] ({m['type']}){marker} {snippet}")
    return "\n".join(out)


def _select_image_evidence(hits: list[dict], max_images: int = 3) -> list[str]:
    """Attach an image only if it is type=image AND in the top 3 (token-cost guard)."""
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


def _answer_anthropic(query: str, hits: list[dict], images: list[str], api_key: str | None = None) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key or config.ANTHROPIC_API_KEY)
    content: list[dict] = [{"type": "text", "text": f"Context:\n{_format_contexts(hits)}"}]
    for p in images:
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": _b64(p)}})
    content.append({"type": "text", "text": f"Question: {query}"})
    resp = client.messages.create(
        model=config.ANTHROPIC_SYNTHESIS_MODEL, max_tokens=1024,
        system=SYSTEM_PROMPT, messages=[{"role": "user", "content": content}])
    return resp.content[0].text.strip()


def _answer_openai(query: str, hits: list[dict], images: list[str], api_key: str | None = None) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key or config.OPENAI_API_KEY)
    content: list[dict] = [{"type": "text", "text": f"Context:\n{_format_contexts(hits)}"}]
    for p in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(p)}"}})
    content.append({"type": "text", "text": f"Question: {query}"})
    resp = client.chat.completions.create(
        model=config.OPENAI_SYNTHESIS_MODEL, max_tokens=1024,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}])
    return resp.choices[0].message.content.strip()


def _answer_groq(query: str, hits: list[dict], images: list[str], api_key: str | None = None) -> tuple[str, dict | None]:
    """JSON mode -> {answer, chart}. Text model, so image *descriptions* in the
    context (not raw bytes) keep answers grounded in visual content."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key or config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL)
    resp = client.chat.completions.create(
        model=config.GROQ_SYNTHESIS_MODEL, max_tokens=1500,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": CHART_SYSTEM_PROMPT},
                  {"role": "user", "content": f"Context:\n{_format_contexts(hits)}\n\nQuestion: {query}"}])
    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
        return data.get("answer", "").strip(), _validate_chart(data.get("chart"))
    except (json.JSONDecodeError, AttributeError):
        return (raw or "").strip(), None


def _answer_extractive(query: str, hits: list[dict], images: list[str]) -> str:
    """No-LLM fallback: return the strongest retrieved passages with citations."""
    return ("(No synthesis LLM configured — showing the most relevant retrieved evidence. "
            "Add an API key for a written answer.)\n\n" + _evidence_lines(hits))


def ask(query: str, k: int = 5, rerank: bool = False, doc_id: str | None = None,
        api_key: str | None = None) -> AnswerResult:
    hits = get_retriever().search(query, k=k, rerank=rerank, doc_id=doc_id)
    images = _select_image_evidence(hits)

    # A user-supplied key (bring-your-own-key) overrides the server default and
    # is used only for this request — never stored.
    user_provider = config.provider_for_key(api_key)
    provider = user_provider or config.SYNTHESIS_PROVIDER
    key = api_key.strip() if user_provider else None

    chart = None
    try:
        if provider == "anthropic":
            text = _answer_anthropic(query, hits, images, key)
        elif provider == "openai":
            text = _answer_openai(query, hits, images, key)
        elif provider == "groq":
            text, chart = _answer_groq(query, hits, images, key)
        else:
            text = _answer_extractive(query, hits, images)
    except Exception as exc:
        # Never 500 on an LLM hiccup — degrade to retrieved evidence + a reason.
        msg = str(exc)
        if "rate_limit" in msg or "429" in msg:
            reason = ("the model's free daily rate limit was reached. Showing retrieved "
                      "evidence instead — try again later, lower **k**, or set a different "
                      "`GROQ_SYNTHESIS_MODEL`.")
        elif "401" in msg or "invalid_api_key" in msg:
            reason = "the API key was rejected. Showing retrieved evidence instead."
        else:
            reason = f"the LLM call failed ({type(exc).__name__}). Showing retrieved evidence instead."
        text = f"⚠️ {reason}\n\n{_evidence_lines(hits)}"
        chart = None
        provider = f"{provider} (fallback)"

    return AnswerResult(
        answer=text, citations=_citations(hits), image_evidence=images,
        provider=provider, chart=chart,
        contexts=[{"page": h["metadata"]["page"], "type": h["metadata"]["type"],
                   "score": round(float(h["score"]), 4)} for h in hits])


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against the indexed corpus.")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()
    res = ask(args.question, k=args.k, rerank=args.rerank)
    print(f"\n=== ANSWER (provider: {res.provider}) ===\n{res.answer}\n")
    print(f"Citations (pages): {res.citations}")
    if res.chart:
        print(f"Chart: {res.chart['type']} — {res.chart['title']}")


if __name__ == "__main__":
    main()
