"""
Luthro — enterprise-style Streamlit UI for the RAG console.
"""

import html
import streamlit as st
import requests
from typing import Dict

# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Luthro · Document Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Fonts + design system (no global * { color } — avoids fighting Streamlit widgets)
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap" rel="stylesheet">
<style>
    :root {
        --l-bg: #f1f5f9;
        --l-bg-subtle: #e8edf3;
        --l-surface: #ffffff;
        --l-elevated: #fafbfc;
        --l-border: #e2e8f0;
        --l-border-strong: #cbd5e1;
        --l-text: #0f172a;
        --l-text-muted: #64748b;
        --l-accent: #1d4ed8;
        --l-accent-hover: #1e40af;
        --l-accent-soft: rgba(29, 78, 216, 0.08);
        --l-success: #059669;
        --l-success-bg: #ecfdf5;
        --l-danger: #dc2626;
        --l-danger-bg: #fef2f2;
        --l-radius: 14px;
        --l-radius-sm: 10px;
        --l-shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 8px 24px rgba(15, 23, 42, 0.06);
        --l-shadow-lg: 0 4px 6px rgba(15, 23, 42, 0.04), 0 24px 48px rgba(15, 23, 42, 0.08);
        --l-font: "Instrument Sans", ui-sans-serif, system-ui, -apple-system, sans-serif;
        --l-serif: "Source Serif 4", Georgia, serif;
    }

    html, body, [data-testid="stAppViewContainer"] {
        font-family: var(--l-font);
        color: var(--l-text);
    }

    .stApp {
        background: linear-gradient(165deg, var(--l-bg) 0%, var(--l-bg-subtle) 45%, #eef2f7 100%);
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    .main .block-container {
        max-width: 1040px;
        margin: 0 auto;
        padding: 1.75rem 1.5rem 3rem;
    }

    /* Hero */
    .luthro-hero {
        text-align: center;
        padding: 2.25rem 1rem 2.5rem;
        margin-bottom: 0.5rem;
    }
    .luthro-hero__brand {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    .luthro-hero__mark {
        width: 40px;
        height: 40px;
        border-radius: 11px;
        background: linear-gradient(145deg, var(--l-accent) 0%, #312e81 100%);
        box-shadow: 0 8px 20px rgba(29, 78, 216, 0.35);
    }
    .luthro-hero__title {
        font-family: var(--l-font);
        font-size: clamp(2rem, 4vw, 2.75rem);
        font-weight: 700;
        letter-spacing: -0.03em;
        color: var(--l-text);
        margin: 0 0 0.5rem 0;
        line-height: 1.15;
    }
    .luthro-hero__title span {
        background: linear-gradient(90deg, #1e40af, #4338ca);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .luthro-hero__tagline {
        font-size: 1.05rem;
        color: var(--l-text-muted);
        font-weight: 500;
        max-width: 32rem;
        margin: 0 auto;
        line-height: 1.55;
    }

    .luthro-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--l-success);
        background: var(--l-surface);
        border: 1px solid rgba(5, 150, 105, 0.25);
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        box-shadow: var(--l-shadow);
    }
    .luthro-pill::before {
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.25);
    }

    /* Section headers (widgets follow in flow — do not wrap in a fake card) */
    .luthro-section-header {
        margin: 1.75rem 0 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--l-border);
    }
    .luthro-section-header:first-of-type { margin-top: 0; }
    .luthro-section__kicker {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--l-accent);
        margin: 0 0 0.25rem 0;
    }
    .luthro-section__title {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--l-text);
        margin: 0;
        letter-spacing: -0.02em;
    }
    .luthro-section__desc {
        font-size: 0.9rem;
        color: var(--l-text-muted);
        margin: 0.35rem 0 0 0;
        line-height: 1.45;
        max-width: 40rem;
    }

    hr.luthro-rule {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--l-border-strong), transparent);
        margin: 0.25rem 0 1.25rem;
    }

    /* Streamlit widgets */
    .stTextInput label,
    .stSelectbox label {
        font-family: var(--l-font) !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        color: var(--l-text) !important;
        text-transform: none;
        letter-spacing: 0;
    }
    .stTextInput input {
        border-radius: var(--l-radius-sm) !important;
        border: 1px solid var(--l-border) !important;
        padding: 0.65rem 0.9rem !important;
        font-size: 0.95rem !important;
        color: var(--l-text) !important;
        background: var(--l-elevated) !important;
        box-shadow: none !important;
    }
    .stTextInput input:focus {
        border-color: var(--l-accent) !important;
        box-shadow: 0 0 0 3px var(--l-accent-soft) !important;
        background: var(--l-surface) !important;
    }
    .stTextInput input::placeholder {
        color: #94a3b8 !important;
    }

    .stSelectbox [data-baseweb="select"] > div {
        border-radius: var(--l-radius-sm) !important;
        border: 1px solid var(--l-border) !important;
        background: var(--l-elevated) !important;
        min-height: 42px;
    }

    /*
     * Selectbox value + list: Streamlit/Baseweb theme often sets white text while we use a
     * light control background — force readable foreground. Popover renders in a portal.
     */
    .stSelectbox [data-baseweb="select"] {
        color: var(--l-text) !important;
    }
    .stSelectbox [data-baseweb="select"] [class*="singleValue"],
    .stSelectbox [data-baseweb="select"] [class*="valueContainer"],
    .stSelectbox [data-baseweb="select"] [class*="placeholder"] {
        color: var(--l-text) !important;
    }
    .stSelectbox [data-baseweb="select"] div,
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] p {
        color: var(--l-text) !important;
    }
    .stSelectbox [data-baseweb="select"] svg {
        fill: var(--l-text-muted) !important;
    }

    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] [role="option"] {
        color: var(--l-text) !important;
        background-color: var(--l-surface) !important;
    }
    div[data-baseweb="popover"] [aria-selected="true"],
    div[data-baseweb="popover"] li[aria-selected="true"] {
        background-color: var(--l-accent-soft) !important;
        color: var(--l-text) !important;
    }

    .row-widget.stHorizontal { gap: 1rem !important; }

    /* Primary buttons */
    .stButton > button {
        font-family: var(--l-font) !important;
        font-weight: 600 !important;
        border-radius: var(--l-radius-sm) !important;
        padding: 0.55rem 1.1rem !important;
        transition: background 0.15s ease, box-shadow 0.15s ease, transform 0.1s ease !important;
    }
    .stButton > button,
    .stButton > button:focus,
    .stButton > button:active {
        background: linear-gradient(180deg, #2563eb 0%, var(--l-accent) 100%) !important;
        color: #fff !important;
        border: 1px solid #1e3a8a !important;
        box-shadow: 0 2px 4px rgba(29, 78, 216, 0.25) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(180deg, #3b82f6 0%, var(--l-accent-hover) 100%) !important;
        box-shadow: 0 4px 14px rgba(29, 78, 216, 0.35) !important;
    }
    .stButton > button p,
    .stButton > button span {
        color: #fff !important;
    }

    /* Results */
    .luthro-results {
        background: var(--l-surface);
        border: 1px solid var(--l-border);
        border-radius: var(--l-radius);
        box-shadow: var(--l-shadow-lg);
        padding: 1.5rem 1.5rem 1.75rem;
        margin-top: 1rem;
    }
    .luthro-results__title {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--l-text);
        margin: 0 0 0.75rem 0;
        font-family: var(--l-font);
    }
    .luthro-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--l-accent);
        background: var(--l-accent-soft);
        border: 1px solid rgba(29, 78, 216, 0.2);
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        margin-bottom: 1rem;
    }
    .luthro-ai {
        font-family: var(--l-serif);
        font-size: 1.05rem;
        line-height: 1.75;
        color: var(--l-text);
        background: linear-gradient(180deg, #f8fafc 0%, #fff 100%);
        border: 1px solid var(--l-border);
        border-left: 4px solid var(--l-accent);
        border-radius: var(--l-radius-sm);
        padding: 1.15rem 1.25rem;
        margin: 0.5rem 0 1.25rem;
    }
    .luthro-inline-h3 {
        font-family: var(--l-font);
        font-size: 1rem;
        font-weight: 700;
        color: var(--l-text);
        margin: 0.25rem 0 0.65rem;
    }
    .luthro-results h3 {
        font-family: var(--l-font);
        font-size: 1rem;
        font-weight: 700;
        color: var(--l-text);
        margin: 1.25rem 0 0.75rem;
    }

    .result-content {
        color: var(--l-text);
        font-size: 0.92rem;
        line-height: 1.65;
        margin: 0.35rem 0 0.75rem;
    }
    .result-metadata {
        color: var(--l-text-muted);
        font-size: 0.85rem;
        line-height: 1.5;
    }
    .result-metadata a {
        color: var(--l-accent) !important;
        font-weight: 600;
        text-decoration: none;
        border-bottom: 1px solid rgba(29, 78, 216, 0.35);
    }
    .result-metadata a:hover {
        color: var(--l-accent-hover) !important;
    }

    .streamlit-expanderHeader {
        background: var(--l-elevated) !important;
        border-radius: var(--l-radius-sm) !important;
        border: 1px solid var(--l-border) !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: var(--l-border-strong) !important;
    }
    .streamlit-expanderHeader p,
    .streamlit-expanderHeader span {
        color: var(--l-text) !important;
    }

    .status-success {
        background: var(--l-success-bg);
        border: 1px solid rgba(5, 150, 105, 0.35);
        color: #065f46;
        padding: 0.85rem 1rem;
        border-radius: var(--l-radius-sm);
        font-weight: 500;
        font-size: 0.92rem;
        text-align: center;
    }
    .status-error {
        background: var(--l-danger-bg);
        border: 1px solid rgba(220, 38, 38, 0.35);
        color: #991b1b;
        padding: 0.85rem 1rem;
        border-radius: var(--l-radius-sm);
        font-weight: 500;
        font-size: 0.92rem;
        text-align: center;
    }

    /* Alerts */
    .stAlert { border-radius: var(--l-radius-sm) !important; }
    [data-testid="stAlert"] p { color: inherit; }

    /* Spinner */
    .stSpinner > div { border-top-color: var(--l-accent) !important; }

    /* Typography helpers */
    .luthro-muted {
        color: var(--l-text-muted);
        font-size: 0.88rem;
        line-height: 1.5;
        margin-top: 0.25rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def check_api_health() -> bool:
    """Check if the API is healthy."""
    try:
        response = requests.get("http://api:8000/healthz", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def ingest_documents(link: str) -> Dict:
    """Ingest documents from Google Drive link."""
    try:
        if "drive.google.com" in link:
            if "/folders/" in link:
                folder_id = link.split("/folders/")[1].split("?")[0]
            else:
                st.error("Please provide a valid Google Drive folder link")
                return {"status": "error"}
        else:
            st.error("Please provide a valid Google Drive folder link")
            return {"status": "error"}

        payload = {"source": "google_drive", "folder_id": folder_id}

        response = requests.post(
            "http://api:8000/ingest",
            json=payload,
            timeout=600,
        )

        if response.status_code == 200:
            return response.json()
        st.error(f"Ingestion failed: {response.text}")
        return {"status": "error"}

    except Exception as e:
        st.error(f"Ingestion error: {str(e)}")
        return {"status": "error"}


def search_documents(query: str, search_mode: str, top_k: int) -> Dict:
    """Search documents using the RAG system."""
    try:
        payload = {
            "question": query,
            "search_mode": search_mode,
            "top_k": top_k,
            "generate_answer": True,
        }

        response = requests.post(
            "http://api:8000/query",
            json=payload,
            timeout=180,
        )

        if response.status_code == 200:
            return response.json()
        st.error(f"Search failed: {response.text}")
        return {"status": "error"}

    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return {"status": "error"}


def main() -> None:
    """Main Luthro application."""

    st.markdown(
        """
        <div class="luthro-hero">
            <div class="luthro-hero__brand">
                <div class="luthro-hero__mark" aria-hidden="true"></div>
            </div>
            <h1 class="luthro-hero__title"><span>Luthro</span></h1>
            <p class="luthro-hero__tagline">
                Document intelligence for teams — ingest from Drive, query with hybrid retrieval, and read grounded answers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not check_api_health():
        st.error(
            "The API service is not available. Start the backend or check your Docker network, then refresh this page."
        )
        st.stop()

    st.markdown(
        '<div style="text-align:center;margin:-0.5rem 0 1.5rem;"><span class="luthro-pill">API connected</span></div>',
        unsafe_allow_html=True,
    )

    # --- Ingestion ---
    st.markdown(
        """
        <div class="luthro-section-header">
            <p class="luthro-section__kicker">01 · Ingestion</p>
            <h2 class="luthro-section__title">Connect your corpus</h2>
            <p class="luthro-section__desc">Paste a shared Google Drive folder link. We index chunks for hybrid retrieval and grounded answers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_link, col_go = st.columns([4, 1], gap="medium")
    with col_link:
        link_input = st.text_input(
            "Google Drive folder URL",
            placeholder="https://drive.google.com/drive/folders/…",
            help="The folder must be readable by the indexer service.",
        )
    with col_go:
        st.markdown("<div style='height:0.15rem'></div>", unsafe_allow_html=True)
        if st.button("Ingest", key="ingest", use_container_width=True):
            if link_input:
                with st.spinner("Indexing documents…"):
                    result = ingest_documents(link_input)
                    if result.get("status") == "success":
                        st.markdown(
                            f'<div class="status-success">Indexed {result.get("chunks_indexed", 0)} chunks from '
                            f'{result.get("documents_processed", 0)} documents.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="status-error">Ingestion could not complete. Verify the folder link and permissions.</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.warning("Enter a Google Drive folder URL first.")

    st.markdown("<hr class='luthro-rule' />", unsafe_allow_html=True)

    # --- Search ---
    st.markdown(
        """
        <div class="luthro-section-header">
            <p class="luthro-section__kicker">02 · Query</p>
            <h2 class="luthro-section__title">Ask your documents</h2>
            <p class="luthro-section__desc">Pick a retrieval mode and how many passages to send to the answer model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    q1, q2, q3 = st.columns([2, 1, 1], gap="medium")
    with q1:
        query = st.text_input(
            "Question",
            placeholder="e.g. What are the main obligations described in the agreement?",
            key="q_main",
        )
    with q2:
        search_mode = st.selectbox(
            "Search mode",
            options=[
                "elser_only",
                "dense_only",
                "bm25_only",
                "dense_bm25",
                "full_hybrid",
            ],
            help="Hybrid modes combine lexical and semantic signals.",
        )
    with q3:
        top_k = st.selectbox(
            "Passages",
            options=[3, 5, 10, 15, 20],
            index=1,
            help="Number of retrieved chunks sent to the answer model.",
        )

    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
    _, c_btn, _ = st.columns([1, 2, 1])
    with c_btn:
        if st.button("Search", key="search", use_container_width=True):
            if query:
                with st.spinner("Retrieving and generating…"):
                    results = search_documents(query, search_mode, top_k)
                    if results.get("status") == "success":
                        st.session_state.search_results = results
                        st.session_state.query = query
                        st.rerun()
                    else:
                        st.error("Search failed. Try again or check API logs.")
            else:
                st.warning("Enter a question to search.")

    # --- Results ---
    if st.session_state.get("search_results"):
        results = st.session_state.search_results
        q_disp = html.escape(str(st.session_state.get("query", "")))

        mode_descriptions = {
            "elser_only": "ELSER semantic",
            "dense_only": "Dense vectors",
            "bm25_only": "BM25 keywords",
            "dense_bm25": "Dense + BM25",
            "full_hybrid": "Full hybrid (ELSER + dense + BM25)",
        }
        mode_label = html.escape(
            str(mode_descriptions.get(results.get("search_mode", ""), results.get("search_mode", "")))
        )

        answer_html = ""
        if results.get("llm_response") and results["llm_response"].get("answer"):
            ans = html.escape(results["llm_response"]["answer"]).replace("\n", "<br/>")
            answer_html = f'<h3 class="luthro-inline-h3">Answer</h3><div class="luthro-ai">{ans}</div>'

        st.markdown(
            f"""
            <div class="luthro-results">
                <p class="luthro-results__title">Results · “{q_disp}”</p>
                <div class="luthro-badge">{mode_label}</div>
                {answer_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if results.get("results"):
            st.markdown(f"### Sources ({len(results['results'])})")
            for i, result in enumerate(results["results"]):
                fn = result.get("filename", "Unknown")
                title = f"{i + 1}. {fn} · score {result.get('_score', 0):.3f}"
                body = html.escape(result.get("content", "No content")).replace("\n", "<br/>")
                with st.expander(title):
                    st.markdown(
                        f'<div class="result-content">{body}</div>',
                        unsafe_allow_html=True,
                    )
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(
                            f'<div class="result-metadata"><strong>File</strong><br>{html.escape(str(result.get("filename", "—")))}</div>',
                            unsafe_allow_html=True,
                        )
                    with m2:
                        st.markdown(
                            f'<div class="result-metadata"><strong>Chunk</strong><br>{html.escape(str(result.get("chunk_id", "—")))}</div>',
                            unsafe_allow_html=True,
                        )
                    with m3:
                        st.markdown(
                            f'<div class="result-metadata"><strong>Type</strong><br>{html.escape(str(result.get("search_type", "—")))}</div>',
                            unsafe_allow_html=True,
                        )
                    if result.get("file_url"):
                        url = html.escape(result.get("file_url", ""), quote=True)
                        st.markdown(
                            f'<div class="result-metadata"><a href="{url}" target="_blank" rel="noopener noreferrer">Open source</a></div>',
                            unsafe_allow_html=True,
                        )

        _padl, clr, _padr = st.columns([2, 1, 2])
        with clr:
            if st.button("Clear results", key="clear", use_container_width=True):
                st.session_state.pop("search_results", None)
                st.session_state.pop("query", None)
                st.rerun()


if __name__ == "__main__":
    main()
