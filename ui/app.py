"""
ui/app.py — Streamlit chat interface (brief §6, Day 6).

A thin client over the FastAPI service: chat on the left, an evidence viewer
(page citations + the actual image evidence) on the right, and PDF upload +
a per-document selector in the sidebar. It holds no RAG logic — every action
is an HTTP call to the API, so the two can be deployed independently.

Visual design follows a "dark-mode data dashboard" system: deep navy surfaces,
a green positive/CTA accent, and Fira Sans / Fira Code typography (monospace
for figures and scores). Run (with the API on :8000):

    streamlit run ui/app.py
"""

from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="Multimodal RAG", page_icon="📊", layout="wide")

# --------------------------------------------------------------------------
# Design system — injected CSS (dark OLED dashboard, green accent, Fira fonts)
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

    :root {
        --bg: #020617; --surface: #0F172A; --surface-2: #1E293B;
        --text: #F8FAFC; --muted: #94A3B8; --accent: #22C55E;
        --border: rgba(148,163,184,0.16);
    }
    html, body, .stApp, [class*="css"] { font-family: 'Fira Sans', sans-serif; }
    .stApp { background:
        radial-gradient(900px 500px at 85% -5%, rgba(34,197,94,0.07), transparent 60%),
        var(--bg); }
    h1, h2, h3, h4 { font-weight: 600; letter-spacing: -0.01em; }
    code, .mono, pre { font-family: 'Fira Code', monospace !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
    [data-testid="stSidebar"] hr { border-color: var(--border); }

    /* Buttons — green CTA, smooth hover, clear pointer */
    .stButton > button, [data-testid="stBaseButton-secondary"] {
        background: var(--accent); color: #04210F; border: 0; border-radius: 10px;
        font-weight: 600; transition: transform .15s ease, filter .15s ease; cursor: pointer;
    }
    .stButton > button:hover { filter: brightness(1.08); }
    .stButton > button:active { transform: translateY(1px); }

    /* Inputs / selects — dark surfaces, green focus ring */
    [data-baseweb="input"], [data-baseweb="select"] > div, .stTextInput input {
        background: var(--surface-2) !important; border-radius: 10px !important;
        border: 1px solid var(--border) !important;
    }
    .stTextInput input:focus, [data-baseweb="select"] > div:focus-within {
        border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(34,197,94,.25) !important;
    }
    *:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }

    /* Chat message cards */
    [data-testid="stChatMessage"] {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 14px; padding: .5rem .9rem; margin-bottom: .5rem;
    }

    /* Reusable pieces rendered via st.markdown */
    .brand { display:flex; align-items:center; gap:.6rem; margin:.2rem 0 1rem; }
    .brand-name { font-size:1.15rem; font-weight:700; }
    .pill { display:inline-flex; align-items:center; gap:.4rem; padding:.18rem .6rem;
        border-radius:999px; font-size:.78rem; font-weight:500; border:1px solid var(--border); }
    .pill.ok  { background:rgba(34,197,94,.12);  color:#86EFAC; border-color:rgba(34,197,94,.3); }
    .pill.err { background:rgba(239,68,68,.12);   color:#FCA5A5; border-color:rgba(239,68,68,.3); }
    .chip { display:inline-block; padding:.12rem .5rem; margin:.12rem .2rem 0 0; border-radius:8px;
        font-family:'Fira Code',monospace; font-size:.72rem; color:var(--muted);
        background:var(--surface-2); border:1px solid var(--border); }
    .cite { display:inline-block; padding:.1rem .5rem; margin:.15rem .25rem 0 0; border-radius:999px;
        font-family:'Fira Code',monospace; font-size:.74rem; color:#86EFAC;
        background:rgba(34,197,94,.1); border:1px solid rgba(34,197,94,.3); }
    .ev-card { background:var(--surface); border:1px solid var(--border); border-radius:12px;
        padding:.55rem .7rem; margin-bottom:.5rem; transition:border-color .2s ease; }
    .ev-card:hover { border-color:rgba(34,197,94,.4); }
    .ev-head { display:flex; justify-content:space-between; font-family:'Fira Code',monospace;
        font-size:.74rem; color:var(--muted); }
    .ev-type { color:#86EFAC; text-transform:uppercase; letter-spacing:.04em; }
    .section-label { font-size:.8rem; font-weight:600; color:var(--muted);
        text-transform:uppercase; letter-spacing:.06em; margin:.2rem 0 .6rem; }

    ::-webkit-scrollbar { width:10px; height:10px; }
    ::-webkit-scrollbar-thumb { background:var(--surface-2); border-radius:8px; }

    @media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# Small inline SVG icons (Lucide-style) — SVG, not emoji, per design guidelines.
ICON_DOC = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#22C55E" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
            '<path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/></svg>')


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def doc_label(doc_id: str, units: int) -> str:
    """Friendly label for the document selector (UUID filenames get shortened)."""
    name = doc_id if len(doc_id) <= 28 else f"{doc_id[:8]}…{doc_id[-4:]}"
    return f"{name}  ·  {units} units"


def api_get(api_url: str, path: str):
    return requests.get(f"{api_url}{path}", timeout=8).json()


# --------------------------------------------------------------------------
# Sidebar — connection, document scope, ingestion, retrieval settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<div class="brand">{ICON_DOC}<span class="brand-name">Multimodal RAG</span></div>',
                unsafe_allow_html=True)

    api_url = st.text_input("API URL", value="http://localhost:8000",
                            label_visibility="collapsed").rstrip("/")

    documents = []
    connected = False
    try:
        info = api_get(api_url, "/")
        documents = api_get(api_url, "/documents").get("documents", [])
        total = sum(d["units"] for d in documents)
        connected = True
        st.markdown(f'<span class="pill ok">● Connected · {total} units indexed</span>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div style="margin-top:.5rem">'
            f'<span class="chip">emb: {info["embedding_provider"]}</span>'
            f'<span class="chip">vision: {info["vision_provider"]}</span>'
            f'<span class="chip">synth: {info["synthesis_provider"]}</span></div>',
            unsafe_allow_html=True)
    except Exception:
        st.markdown('<span class="pill err">● API not reachable</span>', unsafe_allow_html=True)
        st.caption("Start it: `uvicorn api.main:app --port 8000`")

    st.divider()
    st.markdown('<div class="section-label">Search scope</div>', unsafe_allow_html=True)
    # "All documents" -> doc_id=None; otherwise scope to a single document.
    options = {"All documents": None}
    for d in documents:
        options[doc_label(d["doc_id"], d["units"])] = d["doc_id"]
    scope_label = st.selectbox("Document", list(options.keys()), label_visibility="collapsed")
    selected_doc_id = options[scope_label]
    if selected_doc_id:
        st.caption("Answers will use only this document.")

    st.divider()
    st.markdown('<div class="section-label">Add a document</div>', unsafe_allow_html=True)
    pdf = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
    if pdf and st.button("Ingest", use_container_width=True):
        with st.spinner("Parsing, summarizing images, indexing…"):
            try:
                r = requests.post(
                    f"{api_url}/ingest",
                    files={"file": (pdf.name, pdf.getvalue(), "application/pdf")},
                    timeout=600,
                ).json()
                st.success(f"Indexed '{r['doc_id']}' (+{r['units_indexed']} units)")
                st.rerun()
            except Exception as e:
                st.error(f"Ingest failed: {e}")

    st.divider()
    st.markdown('<div class="section-label">Retrieval</div>', unsafe_allow_html=True)
    k = st.slider("Results (k)", 3, 10, 5)
    rerank = st.toggle("Cross-encoder rerank", value=False)

# --------------------------------------------------------------------------
# Main — chat (left) + evidence (right)
# --------------------------------------------------------------------------
st.title("Document Q&A")
scope_note = "across all documents" if selected_doc_id is None else f"in `{scope_label.split('  ·')[0]}`"
st.caption(f"Ask about text, charts, tables, or diagrams — searching {scope_note}. "
           "Answers cite pages and show image evidence.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last" not in st.session_state:
    st.session_state.last = None

chat_col, evidence_col = st.columns([2, 1], gap="large")

with chat_col:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Ask a question about the document…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and answering…"):
                try:
                    resp = requests.post(
                        f"{api_url}/query",
                        json={"query": prompt, "k": k, "rerank": rerank, "doc_id": selected_doc_id},
                        timeout=120,
                    ).json()
                    answer = resp.get("answer", "(no answer)")
                    cites = resp.get("citations", [])
                    pills = "".join(f'<span class="cite">p.{c}</span>' for c in cites)
                    block = answer + (f'<div style="margin-top:.6rem">{pills}</div>' if pills else "")
                    st.markdown(block, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": block})
                    st.session_state.last = resp
                except Exception as e:
                    st.error(f"Query failed: {e}")

with evidence_col:
    st.markdown('<div class="section-label">Evidence</div>', unsafe_allow_html=True)
    last = st.session_state.last
    if not last:
        st.caption("Ask a question to see the retrieved evidence here.")
    else:
        for fn in last.get("image_files", []):
            st.image(f"{api_url}/images/{fn}", use_container_width=True)
        for c in last.get("contexts", []):
            st.markdown(
                f'<div class="ev-card"><div class="ev-head">'
                f'<span class="ev-type">{c["type"]}</span>'
                f'<span>p.{c["page"]} · score {c["score"]}</span></div></div>',
                unsafe_allow_html=True,
            )
