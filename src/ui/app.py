"""
Luthro — enterprise-style Streamlit UI for the RAG console.
"""

import html
import csv
import glob as globlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st

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
    .stSelectbox label,
    .stRadio label,
    .stCheckbox label,
    .stMultiSelect label {
        font-family: var(--l-font) !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        color: var(--l-text) !important;
        text-transform: none;
        letter-spacing: 0;
    }
    .stCheckbox [data-testid="stMarkdownContainer"] p,
    .stRadio [data-testid="stMarkdownContainer"] p {
        color: var(--l-text) !important;
    }
    .stCaption,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    [data-testid="stWidgetLabelHelpInline"] {
        color: var(--l-text-muted) !important;
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

    /* Radio groups (Baseweb) — avoid white-on-light text */
    .stRadio [data-baseweb="radio"] label,
    .stRadio [data-baseweb="radio"] span,
    .stRadio [data-baseweb="radio"] p {
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


def _extract_drive_folder_id(link: str) -> str:
    if "drive.google.com" not in link:
        return ""
    if "/folders/" not in link:
        return ""
    return link.split("/folders/")[1].split("?")[0].strip()


def ingest_documents(
    source: str,
    pipeline_version: str,
    drive_link: str = "",
    file_paths: Optional[List[str]] = None,
) -> Dict:
    """Ingest documents from Google Drive link or local file paths."""
    try:
        payload: Dict[str, object] = {"pipeline_version": pipeline_version}
        if source == "google_drive":
            folder_id = _extract_drive_folder_id(drive_link)
            if not folder_id:
                st.error("Please provide a valid Google Drive folder link")
                return {"status": "error"}
            payload["source"] = "google_drive"
            payload["folder_id"] = folder_id
        elif source == "local_files":
            paths = [p.strip() for p in (file_paths or []) if p.strip()]
            if not paths:
                st.warning("Add at least one local file path, directory, or glob.")
                return {"status": "error"}
            payload["source"] = "local_files"
            payload["file_paths"] = paths
        else:
            st.error(f"Unsupported ingestion source: {source}")
            return {"status": "error"}

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


def search_documents(
    query: str, search_mode: str, top_k: int, pipeline_version: str, generate_answer: bool = True
) -> Dict:
    """Search documents using the RAG system."""
    try:
        payload = {
            "question": query,
            "search_mode": search_mode,
            "top_k": top_k,
            "generate_answer": generate_answer,
            "pipeline_version": pipeline_version,
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


def _default_dataset_path() -> str:
    return os.getenv(
        "BENCHMARK_DATASET_PATH",
        "../kaggle/outputs/pilot_labeled_full_20260417_164339.csv",
    )


def _default_local_ingest_path() -> str:
    return os.getenv("LOCAL_INGEST_PATH", "data/kaggle_docs/*.pdf")


def _read_qrels_from_csv(dataset_path: str, threshold: int = 2) -> List[Dict]:
    path = Path(dataset_path).expanduser()
    if any(ch in dataset_path for ch in "*?[]"):
        matches = sorted(globlib.glob(dataset_path))
        if not matches:
            raise ValueError(f"No dataset files match: {dataset_path}")
        path = Path(matches[-1])
    if not path.exists():
        raise ValueError(f"Dataset file not found: {dataset_path}")

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    required = {"query_idx", "query_text", "doc_name", "passage_id", "relevance_label"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    grouped: Dict[str, Dict] = {}
    for row in rows:
        query_idx = str(row.get("query_idx", "")).strip()
        qid = f"q{query_idx}"
        if qid not in grouped:
            grouped[qid] = {
                "qid": qid,
                "query_idx": query_idx,
                "question": str(row.get("query_text", "")).strip(),
                "relevant": [],
            }
        try:
            relevance = int(float(str(row.get("relevance_label", "0"))))
        except Exception:
            relevance = 0
        if relevance >= threshold:
            doc_name = str(row.get("doc_name", "")).strip()
            passage_id = str(row.get("passage_id", "")).strip()
            grouped[qid]["relevant"].append(f"{doc_name}:{passage_id}")

    return [grouped[k] for k in sorted(grouped.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)]


def _fetch_benchmark_queries(dataset_path: str, threshold: int = 2) -> Tuple[List[Dict], str]:
    try:
        response = requests.get(
            "http://api:8000/benchmark/queries",
            params={"dataset_path": dataset_path, "threshold": threshold},
            timeout=20,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("queries", []), data.get("dataset_path", dataset_path)
    except Exception:
        pass
    qrels = _read_qrels_from_csv(dataset_path, threshold=threshold)
    queries = [
        {
            "qid": q["qid"],
            "query_idx": q.get("query_idx", ""),
            "question": q["question"],
            "relevant_count": len(q.get("relevant", [])),
        }
        for q in qrels
    ]
    return queries, dataset_path


def _fetch_benchmark_qrel(dataset_path: str, qid: str, threshold: int = 2) -> Dict:
    try:
        response = requests.get(
            f"http://api:8000/benchmark/qrels/{qid}",
            params={"dataset_path": dataset_path, "threshold": threshold},
            timeout=20,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("qrel", {})
    except Exception:
        pass
    qrels = _read_qrels_from_csv(dataset_path, threshold=threshold)
    qmap = {q["qid"]: q for q in qrels}
    return qmap.get(qid, {})


def _fetch_benchmark_coverage(dataset_path: str) -> Dict:
    try:
        response = requests.get(
            "http://api:8000/benchmark/coverage",
            params={"dataset_path": dataset_path},
            timeout=20,
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass

    qrels = _read_qrels_from_csv(dataset_path, threshold=2)
    labeled_docs = {r.split(":", 1)[0] for q in qrels for r in q.get("relevant", []) if ":" in r}
    local_docs = {p.stem for p in Path("data/kaggle_docs").glob("*.pdf")}
    missing = sorted(labeled_docs - local_docs)
    return {
        "status": "success",
        "labeled_doc_count": len(labeled_docs),
        "available_doc_count": len(local_docs),
        "missing_docs": missing,
        "coverage_ratio": round(len(labeled_docs & local_docs) / len(labeled_docs), 4) if labeled_docs else 1.0,
    }


def _build_retrieved_ids(results: Dict, pipeline_version: str) -> List[str]:
    out = []
    for r in results.get("results", []):
        if pipeline_version == "v2":
            out.append(f"{r.get('doc_id', '')}:{r.get('page_num', 0)}")
        else:
            filename = str(r.get("filename", ""))
            out.append(f"{Path(filename).stem}:{r.get('chunk_id', 0)}")
    return out


def _benchmark_metrics(relevant: List[str], ranked: List[str], top_k: int) -> Dict[str, float]:
    rel = set(relevant)
    top = ranked[:top_k]
    hit = 1.0 if any(r in rel for r in top) else 0.0
    recall = (sum(1 for r in top if r in rel) / len(rel)) if rel else 0.0
    rr = 0.0
    for i, rid in enumerate(ranked, start=1):
        if rid in rel:
            rr = 1.0 / i
            break
    return {"hit_at_k": hit, "recall_at_k": recall, "mrr": rr}


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
    benchmark_enabled = os.getenv("UI_ENABLE_BENCHMARK_EXPLORER", "false").lower() in {"1", "true", "yes"}

    pipeline_version = st.selectbox(
        "Pipeline",
        options=["v1", "v2"],
        index=0,
        format_func=lambda x: (
            "v1 · Elasticsearch hybrid (ELSER / BM25 / dense)"
            if x == "v1"
            else "v2 · Vision pages (ColPali-class · beta)"
        ),
        help=(
            "v1: text chunks in Elasticsearch. v2: PDF pages as images, vectors in "
            "FAISS/Qdrant — use after ingesting with the same pipeline."
        ),
        key="pipeline_version",
    )
    is_v2 = pipeline_version == "v2"

    # --- Ingestion ---
    ingest_desc = (
        "Choose Google Drive or local files. We index chunks for hybrid retrieval and grounded answers."
        if not is_v2
        else "Choose Google Drive or local files. We rasterize pages, embed them for vision retrieval, then answer via the configured VLM."
    )
    st.markdown(
        f"""
        <div class="luthro-section-header">
            <p class="luthro-section__kicker">01 · Ingestion</p>
            <h2 class="luthro-section__title">Connect your corpus</h2>
            <p class="luthro-section__desc">{html.escape(ingest_desc)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ingest_source = st.selectbox(
        "Ingestion source",
        options=["google_drive", "local_files"],
        format_func=lambda x: "Google Drive" if x == "google_drive" else "Local files",
        help="Local files accepts explicit files, directories, or globs.",
    )

    col_link, col_go = st.columns([4, 1], gap="medium")
    with col_link:
        link_input = ""
        if ingest_source == "google_drive":
            link_input = st.text_input(
                "Google Drive folder URL",
                placeholder="https://drive.google.com/drive/folders/…",
                help="The folder must be readable by the indexer service.",
            )
        else:
            fixed_local_path = _default_local_ingest_path()
            st.caption(
                f"Fixed local corpus path: `{fixed_local_path}`. Click Ingest to index this corpus."
            )
    with col_go:
        st.markdown("<div style='height:0.15rem'></div>", unsafe_allow_html=True)
        if st.button("Ingest", key="ingest", use_container_width=True):
            with st.spinner("Indexing documents…"):
                local_paths = [_default_local_ingest_path()] if ingest_source == "local_files" else []
                result = ingest_documents(
                    source=ingest_source,
                    pipeline_version=pipeline_version,
                    drive_link=link_input,
                    file_paths=local_paths,
                )
                if result.get("status") == "success":
                    if result.get("pipeline_version") == "v2" or is_v2:
                        inner = result.get("result") or {}
                        pages = inner.get("pages_rendered")
                        if pages is None and isinstance(inner.get("totals"), dict):
                            pages = inner["totals"].get("pages_rendered")
                        vecs = inner.get("vectors_indexed")
                        if vecs is None and isinstance(inner.get("totals"), dict):
                            vecs = inner["totals"].get("vectors_indexed")
                        summary = (
                            f"Vision v2 ingest complete. Pages processed: {pages or '—'}, "
                            f"vectors indexed: {vecs or '—'}."
                        )
                        st.markdown(
                            f'<div class="status-success">{html.escape(summary)}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="status-success">Indexed {result.get("chunks_indexed", 0)} chunks from '
                            f'{result.get("documents_processed", 0)} documents.</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        '<div class="status-error">Ingestion could not complete. Verify source inputs and API wiring.</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown("<hr class='luthro-rule' />", unsafe_allow_html=True)

    # --- Search ---
    query_mode = "manual"
    if benchmark_enabled:
        query_mode = st.selectbox(
            "Query mode",
            options=["manual", "benchmark"],
            format_func=lambda x: "Manual question" if x == "manual" else "Benchmark query",
            help="Benchmark mode compares retrieved IDs against labeled relevance.",
        )

    prev_ui_mode = st.session_state.get("_prev_ui_query_mode")
    if benchmark_enabled and prev_ui_mode is not None and prev_ui_mode != query_mode:
        st.session_state.pop("search_results", None)
        st.session_state.pop("benchmark_context", None)
        st.session_state.pop("query", None)
        st.session_state.pop("query_mode", None)

    st.session_state._prev_ui_query_mode = query_mode if benchmark_enabled else "manual"

    query_desc = (
        "Pick a retrieval mode and how many passages to send to the answer model."
        if query_mode == "manual" and not is_v2
        else "Vision retrieval uses page images; top-k controls how many pages are returned."
        if query_mode == "manual"
        else "Select a labeled query and compare retrieved IDs with benchmark ground truth."
    )
    st.markdown(
        f"""
        <div class="luthro-section-header">
            <p class="luthro-section__kicker">02 · Query</p>
            <h2 class="luthro-section__title">Ask your documents</h2>
            <p class="luthro-section__desc">{html.escape(query_desc)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    q1, q2, q3 = st.columns([2, 1, 1], gap="medium")
    selected_qrel: Dict = {}
    benchmark_coverage: Dict = {}
    selected_qid = ""
    with q1:
        if query_mode == "manual":
            query = st.text_input(
                "Question",
                placeholder="e.g. What are the main obligations described in the agreement?",
                key="q_main",
            )
        else:
            dataset_path = _default_dataset_path()
            try:
                queries, resolved_path = _fetch_benchmark_queries(dataset_path)
                st.caption(
                    f"Benchmark dataset loaded ({len(queries)} queries) from fixed path: {resolved_path}"
                )
            except Exception as e:
                queries, resolved_path = [], dataset_path
                st.error(f"Failed to load benchmark queries: {e}")
            selected_qid = st.selectbox(
                "Labeled query",
                options=[q.get("qid", "") for q in queries],
                format_func=lambda qid: next(
                    (
                        f"{qid} · {q.get('question', '')[:90]}"
                        for q in queries
                        if q.get("qid") == qid
                    ),
                    qid,
                ),
            ) if queries else ""
            selected_qrel = _fetch_benchmark_qrel(resolved_path, selected_qid) if selected_qid else {}
            benchmark_coverage = _fetch_benchmark_coverage(resolved_path) if queries else {}
            query = selected_qrel.get("question", "")
    with q2:
        if is_v2:
            st.caption("Hybrid search modes apply to **v1** only.")
            search_mode = "dense_bm25"
        else:
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
            "Passages" if not is_v2 else "Pages (top-k)",
            options=[3, 5, 10, 15, 20],
            index=1,
            help=(
                "v1: chunks sent to the LLM. v2: page images retrieved and sent to the VLM."
            ),
        )

    benchmark_generate_answer = False
    if benchmark_enabled and query_mode == "benchmark":
        benchmark_generate_answer = st.checkbox(
            "Generate answer (benchmark)",
            value=False,
            help="Off by default for retrieval benchmarking; enable when you want the LLM/VLM answer too.",
        )

    st.session_state.query_mode = query_mode

    if benchmark_enabled and query_mode == "benchmark":
        bctx_prev = st.session_state.get("benchmark_context") or {}
        prev_qid = str(bctx_prev.get("qid", "") or "")
        if prev_qid and selected_qid and prev_qid != selected_qid:
            st.session_state.pop("search_results", None)
            st.session_state.pop("benchmark_context", None)
        st.session_state.query = query

    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
    _, c_btn, _ = st.columns([1, 2, 1])
    with c_btn:
        button_text = "Search" if query_mode == "manual" else "Run benchmark query"
        if st.button(button_text, key="search", use_container_width=True):
            if query:
                gen_ans = (query_mode == "manual") or bool(benchmark_generate_answer)
                spin_text = (
                    "Retrieving and generating…"
                    if gen_ans
                    else ("Running benchmark retrieval…" if query_mode == "benchmark" else "Retrieving…")
                )
                with st.spinner(spin_text):
                    results = search_documents(
                        query, search_mode, top_k, pipeline_version, generate_answer=gen_ans
                    )
                    if results.get("status") == "success":
                        st.session_state.search_results = results
                        st.session_state.query = query
                        st.session_state.query_mode = query_mode
                        if query_mode == "benchmark":
                            ranked_ids = _build_retrieved_ids(results, pipeline_version)
                            relevant = selected_qrel.get("relevant", [])
                            st.session_state.benchmark_context = {
                                "dataset_path": _default_dataset_path(),
                                "qid": selected_qrel.get("qid", ""),
                                "question": selected_qrel.get("question", query),
                                "top_k": top_k,
                                "relevant": relevant,
                                "ranked": ranked_ids,
                                "coverage": benchmark_coverage,
                                "metrics": _benchmark_metrics(relevant, ranked_ids, top_k),
                            }
                        st.rerun()
                    else:
                        st.error("Search failed. Try again or check API logs.")
            else:
                st.warning("Enter a question to search." if query_mode == "manual" else "Select a benchmark query first.")

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
            "vision_colpali_faiss": "Vision v2 (page vectors)",
        }
        sm = results.get("search_mode", "")
        mode_label = html.escape(
            str(mode_descriptions.get(sm, sm or "—"))
        )
        pv_badge = html.escape(str(results.get("pipeline_version", "v1")))

        answer_html = ""
        if results.get("llm_response") and results["llm_response"].get("answer"):
            ans = html.escape(results["llm_response"]["answer"]).replace("\n", "<br/>")
            answer_html = f'<h3 class="luthro-inline-h3">Answer</h3><div class="luthro-ai">{ans}</div>'

        st.markdown(
            f"""
            <div class="luthro-results">
                <p class="luthro-results__title">Results · “{q_disp}”</p>
                <div class="luthro-badge">Pipeline {pv_badge}</div>
                <div class="luthro-badge">{mode_label}</div>
                {answer_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.get("query_mode") == "benchmark":
            bctx = st.session_state.get("benchmark_context", {})
            b_top_k = int(bctx.get("top_k", top_k))
            metrics = bctx.get("metrics", {})
            coverage = bctx.get("coverage", {})
            st.markdown("### Benchmark Comparator")
            if not bctx.get("relevant"):
                st.info(
                    "Ground-truth labels are unavailable for this benchmark query, so Recall/MRR/Hit metrics cannot be computed."
                )
            else:
                st.caption(
                    f"QID {bctx.get('qid', '—')} · Recall@{b_top_k}: {metrics.get('recall_at_k', 0.0):.3f} · "
                    f"MRR: {metrics.get('mrr', 0.0):.3f} · Hit@{b_top_k}: {int(metrics.get('hit_at_k', 0.0))}"
                )
            missing_docs = coverage.get("missing_docs", []) if isinstance(coverage, dict) else []
            if missing_docs:
                st.warning(
                    f"Partial corpus detected: {len(missing_docs)} labeled docs are missing locally. "
                    "Benchmark scores may be lower than full-corpus evaluation."
                )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Ground truth IDs**")
                st.code("\n".join(bctx.get("relevant", [])) or "(none)", language="text")
            with c2:
                st.markdown("**Retrieved IDs**")
                st.code("\n".join(bctx.get("ranked", [])) or "(none)", language="text")

        if results.get("timings_ms"):
            tm = results["timings_ms"]
            st.caption(
                f"Timings (ms): retrieve {tm.get('retrieve', '—')} · "
                f"generate {tm.get('generate', '—')}"
            )

        if results.get("results"):
            st.markdown(f"### Sources ({len(results['results'])})")
            for i, result in enumerate(results["results"]):
                fn = result.get("filename", "Unknown")
                title = f"{i + 1}. {fn} · score {result.get('_score', 0):.3f}"
                body_raw = result.get("content", "").strip()
                if body_raw:
                    body = html.escape(body_raw).replace("\n", "<br/>")
                else:
                    body = (
                        "<em>Vision page — see metadata and preview below.</em>"
                        if result.get("image_path")
                        else "<em>No text chunk for this hit.</em>"
                    )
                with st.expander(title):
                    st.markdown(
                        f'<div class="result-content">{body}</div>',
                        unsafe_allow_html=True,
                    )
                    ip = result.get("image_path") or ""
                    if ip:
                        try:
                            p = Path(str(ip))
                            if p.is_file():
                                st.image(str(p), caption=f"Page {result.get('page_num', '—')}")
                        except Exception:
                            pass
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(
                            f'<div class="result-metadata"><strong>File</strong><br>{html.escape(str(result.get("filename", "—")))}</div>',
                            unsafe_allow_html=True,
                        )
                    with m2:
                        if result.get("page_num") is not None:
                            chunk_or_page = html.escape(
                                f"page {result.get('page_num', '—')}"
                            )
                            label = "Page"
                        else:
                            chunk_or_page = html.escape(
                                str(result.get("chunk_id", "—"))
                            )
                            label = "Chunk"
                        st.markdown(
                            f'<div class="result-metadata"><strong>{html.escape(label)}</strong><br>{chunk_or_page}</div>',
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
                st.session_state.pop("benchmark_context", None)
                st.session_state.pop("query_mode", None)
                st.rerun()


if __name__ == "__main__":
    main()
