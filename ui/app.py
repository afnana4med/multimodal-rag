"""
ui/app.py — Streamlit chat interface (brief §6, Day 6).

A thin client over the FastAPI service: chat on the left, an evidence viewer
(page citations + the actual image evidence) on the right, and PDF upload in
the sidebar. It holds no RAG logic — every action is an HTTP call to the API,
which means you can point this at a locally-running API or a deployed one.

Run (with the API already running on :8000):
    streamlit run ui/app.py
"""

from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="Multimodal RAG", layout="wide")

# --- Sidebar: connection, ingestion, retrieval settings -------------------
with st.sidebar:
    st.title("⚙️ Settings")
    api_url = st.text_input("API URL", value="http://localhost:8000").rstrip("/")

    # Show live service config / index size.
    try:
        info = requests.get(f"{api_url}/", timeout=5).json()
        n = requests.get(f"{api_url}/stats", timeout=5).json().get("count", 0)
        st.success(f"Connected · {n} units indexed")
        st.caption(
            f"embeddings: {info['embedding_provider']} · "
            f"vision: {info['vision_provider']} · synth: {info['synthesis_provider']}"
        )
    except Exception:
        st.error("API not reachable. Start it: `uvicorn api.main:app --port 8000`")

    st.divider()
    st.subheader("📄 Add a document")
    pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if pdf and st.button("Ingest", use_container_width=True):
        with st.spinner("Parsing, summarizing images, indexing..."):
            try:
                r = requests.post(
                    f"{api_url}/ingest",
                    files={"file": (pdf.name, pdf.getvalue(), "application/pdf")},
                    timeout=600,
                )
                res = r.json()
                st.success(f"Indexed '{res['doc_id']}' (+{res['units_indexed']} units)")
            except Exception as e:
                st.error(f"Ingest failed: {e}")

    st.divider()
    k = st.slider("Results (k)", 3, 10, 5)
    rerank = st.toggle("Cross-encoder rerank", value=False)

# --- Main: chat (left) + evidence (right) ---------------------------------
st.title("📊 Multimodal RAG — Document Q&A")
st.caption("Ask about text, charts, tables, or diagrams. Answers cite pages and show image evidence.")

if "messages" not in st.session_state:
    st.session_state.messages = []   # [{role, content}]
if "last" not in st.session_state:
    st.session_state.last = None      # last API response (for evidence panel)

chat_col, evidence_col = st.columns([2, 1])

with chat_col:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and answering..."):
                try:
                    resp = requests.post(
                        f"{api_url}/query",
                        json={"query": prompt, "k": k, "rerank": rerank},
                        timeout=120,
                    ).json()
                    answer = resp.get("answer", "(no answer)")
                    cites = resp.get("citations", [])
                    suffix = f"\n\n*Pages: {', '.join(map(str, cites))}*" if cites else ""
                    st.markdown(answer + suffix)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer + suffix})
                    st.session_state.last = resp
                except Exception as e:
                    st.error(f"Query failed: {e}")

with evidence_col:
    st.subheader("🔎 Evidence")
    last = st.session_state.last
    if not last:
        st.info("Ask a question to see the retrieved evidence here.")
    else:
        for fn in last.get("image_files", []):
            st.image(f"{api_url}/images/{fn}", caption=fn, use_container_width=True)
        st.markdown("**Retrieved units**")
        for c in last.get("contexts", []):
            st.markdown(f"- `{c['type']}` p.{c['page']} · score {c['score']}")
