# Multimodal RAG — Document Q&A with Chart & Table Understanding

> **Standard RAG drops 30–40% of the information in real business documents** because charts, tables, and diagrams are invisible to text-only retrieval. This system uses vision models to make that content searchable — so *"what does Figure 1 show?"* becomes a valid query, not a dead end.

A Retrieval-Augmented Generation system that answers questions about complex PDFs (financial reports, research papers, technical manuals) by understanding **both text and visual content**. When you ask a question, it retrieves the most relevant evidence — whether it lives in a paragraph, a bar chart, or a table — and returns a grounded answer with **page citations** and the **actual image evidence** embedded.

---

## Why this exists

Most RAG pipelines treat a PDF as a flat stream of text: extract paragraphs, chunk, embed, done. That works for blog posts and breaks on the documents people actually query at work, where critical information is trapped inside:

- **Charts** — revenue trends, growth rates, market share
- **Tables** — financial statements, comparison matrices
- **Diagrams** — architectures, process flows
- **Figures with embedded text** — annotated plots, infographics

This project solves it with three ideas:

1. **Layout-aware extraction** — separate a PDF into `text`, `image`, and `table` elements, each tagged with its page number (for citations).
2. **Vision summarization** — each image is described by a vision model; that description is embedded *alongside* text, so images become searchable by meaning (not by visual similarity, which is what CLIP does badly).
3. **Image evidence in synthesis** — when retrieval surfaces an image, the *actual* image is attached to the answering LLM call, so the answer is grounded in the real pixels.

---

## Architecture

```
PDF
 │  PyMuPDF + pdfplumber
 ▼
text / image / table elements  (each tagged with page #)
 │                       │
 │                  vision summary  (Claude / GPT-4o / OCR)  →  searchable text
 ▼                       ▼
        one embedding model for BOTH  (text-embedding-3-small or bge-small)
 │
 ▼
ChromaDB  ── single mixed-modality collection (cosine) ──┐
 │                                                        │
 ▼                                                        │
Hybrid retrieval:  vector  +  BM25  →  Reciprocal Rank Fusion
 │                                  (+ optional cross-encoder rerank)
 ▼
Answer synthesis  (top-3 images attached to the LLM call)
 │
 ▼
{ answer, page citations, image evidence }
```

### Provider-agnostic by design

No vendor is hardcoded. Each stage auto-detects API keys from `.env` and **upgrades automatically** — with zero code changes:

| Stage | Free / local (default) | Upgraded (with a key) |
|---|---|---|
| Embeddings | `sentence-transformers` (bge-small) | OpenAI `text-embedding-3-small` |
| Vision (image→text) | `tesseract` OCR | Claude `claude-sonnet-4-6` / GPT-4o |
| Synthesis | extractive (top passages) | Claude / GPT-4o (writes the answer) |

This means the whole system **runs end-to-end for $0**, and you add capability incrementally by dropping a key into `.env`.

---

## Results

Measured on a ground-truth eval set (`tests/eval_queries.json`) via `tests/eval.py`:

| Metric | Result | Target |
|---|---|---|
| **Retrieval precision @5** | **100%** | >80% |
| **Modality routing** (image surfaced when expected) | **90%** | — |
| Answer-contains (free/extractive mode) | 60% | >70%¹ |

¹ *Answer-contains is measured in free extractive mode, where the "answer" is the raw retrieved passages (so a fact can fall outside the snippet window even when the correct page was retrieved). Adding a synthesis key lets an LLM read the full context and raises this metric.*

---

## Quickstart

### 1. System dependency
```bash
brew install tesseract          # macOS  (Ubuntu: sudo apt install tesseract-ocr)
```

### 2. Python env + deps
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. (Optional) add API keys for higher quality
```bash
cp .env.example .env            # then uncomment/fill ANTHROPIC_API_KEY or OPENAI_API_KEY
```
Without keys it runs fully local/free.

### 4. Ingest a PDF
```bash
python ingestion/parse_pdf.py data/pdfs/your.pdf        # parse -> elements + images
python ingestion/summarize_images.py data/pdfs/your.pdf # images -> searchable text
python indexing/store.py data/pdfs/your.pdf             # chunk -> embed -> index
```

### 5. Ask a question
```bash
python synthesis/answerer.py "What was revenue in Q4 and how did it trend?"
```

### 6. Run the API + UI
```bash
uvicorn api.main:app --reload --port 8000      # terminal 1
streamlit run ui/app.py                         # terminal 2
```

### 7. Evaluate
```bash
python tests/eval.py --k 5
```

---

## Project structure

```
ingestion/   parse_pdf.py · summarize_images.py · chunk_text.py
indexing/    embed.py · store.py
retrieval/   retriever.py (vector+BM25+RRF) · reranker.py
synthesis/   answerer.py (multimodal answer w/ image evidence)
api/         main.py (FastAPI: /ingest, /query, /stats, /images)
ui/          app.py (Streamlit chat + evidence viewer)
tests/       eval_queries.json · eval.py
config.py    central provider/path config
```

---

## Key engineering decisions

- **PyMuPDF + pdfplumber over `unstructured` hi-res.** Same multimodal output (text/image/table + page numbers) without a multi-GB ML layout model — fast, Python-3.13-friendly, and deployable on a free tier. Real-world parser noise (figure grids misdetected as tables) is handled with a column-cap + content-quality filter.
- **One embedding model for everything.** Text chunks and image summaries *must* share an embedding space or cross-modal similarity is meaningless — enforced by routing all content through a single embedder, with the model name stored in each record's metadata.
- **Hybrid retrieval (vector + BM25 + RRF).** Vector search matches meaning; BM25 catches exact tokens ("Figure 3", "WMT 2014", "$89B"). RRF fuses them on rank, so neither blind spot dominates.
- **Cost-guarded image evidence.** Images are attached to the LLM only when a hit is both type `image` and in the top 3 — visual grounding without runaway token cost.

---

## What I learned

- **The hard part of multimodal RAG isn't the LLM — it's the ingestion.** Getting clean text/image/table separation out of messy PDFs, and filtering parser false-positives, is where most of the real engineering lives.
- **Vision-LLM summarization beats image embeddings (CLIP) for Q&A.** CLIP finds images that *look* alike; describing-then-embedding finds images that *answer the question*.
- **Provider abstraction pays off immediately.** Building free-local + hosted paths behind one interface meant the system was demoable from day one and upgradeable without rewrites.
- **Measure retrieval first.** If the right page isn't in the top-k, no synthesis trick saves you. Precision@k is the metric that matters most.

---

*Built as a portfolio project. Lead use case: financial-report Q&A (10-Ks, earnings decks), with the same architecture applying to legal, healthcare, consulting, and engineering documents.*
