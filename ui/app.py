"""Streamlit chat UI for the multimodal RAG system. Thin client over the API."""

from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="Multimodal RAG", page_icon="📊",
                   layout="centered", initial_sidebar_state="expanded")

CHART_PALETTE = ["#4F46E5", "#06B6D4", "#22C55E", "#F59E0B", "#EF4444", "#A855F7"]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    :root { --accent:#4F46E5; --text:#0F172A; --muted:#64748B; --border:#E6E8EC; --surface:#F7F8FA; }
    html, body, .stApp, [class*="css"] { font-family:'Inter',sans-serif; }
    .stApp { background:#FFFFFF; }
    #MainMenu, header[data-testid="stHeader"], footer { display:none; }
    .block-container { max-width:780px; padding-top:2.2rem; padding-bottom:7rem; }
    h1,h2,h3 { letter-spacing:-0.02em; font-weight:600; }
    [data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--border); }
    [data-testid="stChatMessage"] { background:transparent; padding:.35rem 0; gap:.7rem; }
    [data-testid="stChatMessageContent"] { font-size:0.97rem; line-height:1.7; color:var(--text); }
    [data-testid="stChatInput"] { border-radius:16px !important; border:1px solid var(--border) !important;
        box-shadow:0 6px 24px rgba(15,23,42,0.06) !important; }
    [data-testid="stChatInput"]:focus-within { border-color:var(--accent) !important;
        box-shadow:0 0 0 3px rgba(79,70,229,.15) !important; }
    button[data-testid="stBaseButton-primary"]{ background:var(--accent)!important; border:0!important;
        border-radius:12px!important; font-weight:600!important; }
    button[data-testid="stBaseButton-primary"]:hover{ filter:brightness(1.07); }
    button[data-testid="stBaseButton-secondary"]{ background:#fff!important; color:var(--text)!important;
        border:1px solid var(--border)!important; border-radius:12px!important; font-weight:500!important;
        text-align:left!important; transition:all .15s ease; }
    button[data-testid="stBaseButton-secondary"]:hover{ border-color:var(--accent)!important;
        background:#FAFAFF!important; transform:translateY(-1px); }
    .stButton>button{ cursor:pointer; }
    *:focus-visible{ outline:2px solid var(--accent)!important; outline-offset:2px; }
    .hero { text-align:center; margin:2.2rem 0 1.4rem; }
    .hero-title { font-size:2rem; font-weight:600; letter-spacing:-0.03em; color:var(--text); }
    .hero-sub { color:var(--muted); font-size:1.0rem; margin-top:.6rem; line-height:1.6; }
    .cites { margin-top:.5rem; }
    .cite { display:inline-block; padding:.1rem .55rem; margin:.15rem .3rem 0 0; border-radius:999px;
        font-size:.74rem; font-weight:500; color:var(--accent); background:#EEF2FF; border:1px solid #E0E4FF; }
    .pill { display:inline-flex; align-items:center; gap:.35rem; padding:.18rem .6rem; border-radius:999px;
        font-size:.76rem; font-weight:500; border:1px solid var(--border); }
    .pill.ok{ background:#ECFDF5; color:#047857; border-color:#A7F3D0; }
    .pill.err{ background:#FEF2F2; color:#B91C1C; border-color:#FECACA; }
    .chip-tag{ display:inline-block; padding:.1rem .45rem; margin:.1rem .2rem 0 0; border-radius:7px;
        font-size:.7rem; color:var(--muted); background:#fff; border:1px solid var(--border); }
    .slabel{ font-size:.74rem; font-weight:600; color:var(--muted); text-transform:uppercase;
        letter-spacing:.06em; margin:.2rem 0 .5rem; }
    @media (prefers-reduced-motion: reduce){ *{ transition:none!important; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=8, show_spinner=False)
def fetch_status(api_url: str):
    """Service info + indexed documents (cached briefly to avoid refetching on every rerun)."""
    info = requests.get(f"{api_url}/", timeout=6).json()
    docs = requests.get(f"{api_url}/documents", timeout=6).json().get("documents", [])
    return info, docs


def ingest_pdf(api_url: str, name: str, data: bytes, api_key: str | None = None):
    return requests.post(f"{api_url}/ingest",
                         files={"file": (name, data, "application/pdf")},
                         data={"api_key": api_key} if api_key else None, timeout=600).json()


def key_provider(key: str | None) -> str | None:
    """Friendly provider name from a key prefix (for the bring-your-own-key field)."""
    k = (key or "").strip()
    if k.startswith("gsk_"):
        return "Groq"
    if k.startswith("sk-ant-"):
        return "Claude"
    if k.startswith("sk-"):
        return "OpenAI"
    return None


def render_chart(spec: dict) -> None:
    import plotly.graph_objects as go

    ctype, cats, series = spec["type"], spec["categories"], spec["series"]
    fig = go.Figure()
    if ctype == "pie":
        fig.add_trace(go.Pie(labels=cats, values=series[0]["values"], hole=0.5,
                             marker=dict(colors=CHART_PALETTE), textinfo="label+percent"))
    elif ctype == "line":
        for i, s in enumerate(series):
            fig.add_trace(go.Scatter(x=cats, y=s["values"], mode="lines+markers", name=s["name"],
                          line=dict(color=CHART_PALETTE[i % len(CHART_PALETTE)], width=3)))
    else:
        for i, s in enumerate(series):
            fig.add_trace(go.Bar(x=cats, y=s["values"], name=s["name"],
                          marker_color=CHART_PALETTE[i % len(CHART_PALETTE)],
                          text=[f"{v:,.0f}" for v in s["values"]], textposition="outside"))
        fig.update_layout(barmode="group")
    fig.update_layout(
        title=dict(text=spec.get("title", ""), font=dict(size=15, color="#0F172A")),
        xaxis_title=spec.get("x_label"), yaxis_title=spec.get("y_label"),
        template="plotly_white", font=dict(family="Inter", size=12, color="#0F172A"),
        margin=dict(t=48, l=8, r=8, b=8), height=380, showlegend=len(series) > 1,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_message(m: dict) -> None:
    with st.chat_message(m["role"]):
        if m.get("content"):
            st.markdown(m["content"], unsafe_allow_html=True)
        if m.get("chart"):
            render_chart(m["chart"])
        for url in m.get("images", []):
            st.image(url, use_container_width=True)
        if m.get("citations"):
            pills = "".join(f'<span class="cite">p.{c}</span>' for c in m["citations"])
            st.markdown(f'<div class="cites">{pills}</div>', unsafe_allow_html=True)


def doc_label(doc_id: str, units: int) -> str:
    name = doc_id if len(doc_id) <= 26 else f"{doc_id[:8]}…{doc_id[-4:]}"
    return f"{name}  ·  {units} units"


st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)

# --- Sidebar: connection, document scope, upload, settings ---
with st.sidebar:
    st.markdown("### 📊 Multimodal RAG")
    api_url = st.text_input("API URL", value="http://localhost:8000",
                            label_visibility="collapsed").rstrip("/")
    documents = []
    try:
        info, documents = fetch_status(api_url)
        total = sum(d["units"] for d in documents)
        st.markdown(f'<span class="pill ok">● Connected · {total} units</span>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<span class="pill err">● API not reachable</span>', unsafe_allow_html=True)
        st.caption("Start it: `./run.sh`")

    st.divider()
    st.markdown('<div class="slabel">Your API key</div>', unsafe_allow_html=True)
    user_key = st.text_input("API key", type="password", label_visibility="collapsed",
                             placeholder="gsk_… (Groq) / sk-… / sk-ant-…").strip()
    kp = key_provider(user_key)
    if kp:
        st.markdown(f'<span class="pill ok">● using your {kp} key</span>', unsafe_allow_html=True)
    elif user_key:
        st.markdown('<span class="pill err">● unrecognized key</span>', unsafe_allow_html=True)
    else:
        st.caption("Paste a free [Groq](https://console.groq.com) key (or OpenAI / Claude) "
                   "for written answers + charts. Used per request, never stored.")

    st.divider()
    st.markdown('<div class="slabel">Document</div>', unsafe_allow_html=True)
    options = {"All documents": None}
    for d in documents:
        options[doc_label(d["doc_id"], d["units"])] = d["doc_id"]
    scope = st.selectbox("Scope", list(options.keys()),
                         index=1 if len(options) == 2 else 0, label_visibility="collapsed")
    selected_doc_id = options[scope]

    st.divider()
    k = st.slider("Results (k)", 3, 10, 6)
    rerank = st.toggle("Cross-encoder rerank", value=False)
    if st.button("＋ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


def upload_widget(key: str) -> None:
    """File uploader available both in the sidebar and the main welcome area, so
    it never disappears when the sidebar is collapsed."""
    pdf = st.file_uploader("Upload a PDF", type=["pdf"], key=key, label_visibility="collapsed")
    if pdf and st.button("Ingest document", type="primary", key=f"btn_{key}", use_container_width=True):
        with st.spinner("Parsing, summarizing, indexing…"):
            try:
                r = ingest_pdf(api_url, pdf.name, pdf.getvalue(), user_key or None)
                fetch_status.clear()
                st.success(f"Indexed '{r['doc_id']}' (+{r['units_indexed']})")
                st.rerun()
            except Exception as e:
                st.error(f"Ingest failed: {e}")


with st.sidebar:
    st.markdown('<div class="slabel">Add a document</div>', unsafe_allow_html=True)
    upload_widget("up_side")

# --- Resolve prompt (chat input or a clicked suggestion chip) ---
typed = st.chat_input("Ask anything about your document…")
prompt = typed or st.session_state.pop("pending", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        with st.spinner("Thinking…"):
            resp = requests.post(
                f"{api_url}/query",
                json={"query": prompt, "k": k, "rerank": rerank,
                      "doc_id": selected_doc_id, "api_key": user_key or None},
                timeout=120,
            ).json()
        st.session_state.messages.append({
            "role": "assistant",
            "content": resp.get("answer", "(no answer)"),
            "chart": resp.get("chart"),
            "images": [f"{api_url}/images/{fn}" for fn in resp.get("image_files", [])],
            "citations": resp.get("citations", []),
        })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ Request failed: {e}\n\nIs the API running? Start it with `./run.sh`.",
        })
    st.rerun()

# --- Welcome state or conversation ---
if not st.session_state.messages:
    st.markdown(
        '<div class="hero"><div class="hero-title">What would you like to know?</div>'
        '<div class="hero-sub">Ask about any uploaded document — text, tables, or figures.<br>'
        'I can retrieve figures from the file and build charts from its data.</div></div>',
        unsafe_allow_html=True,
    )
    with st.expander("📎 Upload a document", expanded=not documents):
        upload_widget("up_main")
    suggestions = [
        "Summarize this report",
        "Chart revenue by product",
        "Show net sales by region as a bar chart",
        "What were net income and gross margin?",
    ]
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, key=f"chip{i}", type="secondary", use_container_width=True):
            st.session_state.pending = s
            st.rerun()
else:
    for m in st.session_state.messages:
        render_message(m)
