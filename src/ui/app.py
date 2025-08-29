"""
Luthro - Advanced RAG Search System
Clean, modern design with light blue theme and professional search components.
"""

import streamlit as st
import requests
import time
from typing import Dict, List, Optional
import json

# Page configuration
st.set_page_config(
    page_title="Luthro - AI Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced Luthro styling
st.markdown("""
<style>
    /* Global background */
    .main {
        background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 50%, #e0f7fa 100%);
        min-height: 100vh;
    }
    
    /* Header styling with better alignment and contrast */
    .luthro-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin: 0 auto 1rem auto;
        color: #1e293b;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        display: block;
        width: 100%;
    }
    
    .luthro-subtitle {
        text-align: center;
        color: #475569;
        font-size: 1.2rem;
        margin: 0 auto 1.5rem auto;
        font-weight: 500;
        display: block;
        width: 100%;
    }
    
    /* Container styling with better alignment and contrast */
    .luthro-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 0 auto 2rem auto;
        width: 100%;
        box-sizing: border-box;
    }
    
    .link-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 2px dashed #0ea5e9;
        margin: 0 auto 1rem auto;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.1);
        width: 100%;
        box-sizing: border-box;
    }
    
    .search-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(14, 165, 233, 0.1);
        margin: 0 auto 2rem auto;
        width: 100%;
        box-sizing: border-box;
    }
    
    .results-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(14, 165, 233, 0.1);
        width: 100%;
        box-sizing: border-box;
    }
    
    /* Enhanced search bar styling with better alignment and contrast */
    .stTextInput {
        width: 100%;
        margin-bottom: 1rem;
    }
    
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 1rem 1.5rem;
        font-size: 1.1rem;
        background: white;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        width: 100%;
        box-sizing: border-box;
        color: #334155;
        caret-color: #0ea5e9;
        outline: none;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #0ea5e9;
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1);
        outline: none;
        background: #ffffff;
        caret-color: #0ea5e9;
    }
    
    .stTextInput > div > div > input:hover {
        border-color: #0284c7;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.15);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #94a3b8;
    }
    
    /* Enhanced dropdown styling with better alignment and contrast */
    .stSelectbox {
        width: 100%;
        margin-bottom: 1rem;
    }
    
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        background: white;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        width: 100%;
        box-sizing: border-box;
        color: #334155;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #0284c7;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.15);
    }
    
    /* Button styling with better alignment and contrast */
    .luthro-button-primary {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.3);
        width: 100%;
        box-sizing: border-box;
    }
    
    .luthro-button-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(14, 165, 233, 0.4);
    }
    
    .luthro-button-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
        width: 100%;
        box-sizing: border-box;
    }
    
    .luthro-button-success:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
    }
    
    .luthro-button-secondary {
        background: linear-gradient(135deg, #64748b 0%, #475569 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(100, 116, 139, 0.2);
        width: 100%;
        box-sizing: border-box;
    }
    
    .luthro-button-secondary:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(100, 116, 139, 0.3);
    }
    
    /* Result card styling with better alignment and contrast */
    .result-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 4px solid #0ea5e9;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        width: 100%;
        box-sizing: border-box;
        color: #334155;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        line-height: 1.6;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* AI answer styling with better alignment and contrast */
    .ai-answer {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #0ea5e9;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1);
        width: 100%;
        box-sizing: border-box;
        color: #1e293b;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 1rem;
        line-height: 1.7;
        font-weight: 400;
    }
    
    /* Mode badge styling with better alignment and contrast */
    .mode-badge {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin: 0 auto 1rem auto;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.2);
        text-align: center;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Progress and status styling with better alignment and contrast */
    .status-success {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 1px solid #22c55e;
        color: #166534;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem auto;
        width: 100%;
        box-sizing: border-box;
        text-align: center;
        font-weight: 500;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    .status-error {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        border: 1px solid #ef4444;
        color: #991b1b;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem auto;
        width: 100%;
        box-sizing: border-box;
        text-align: center;
        font-weight: 500;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    /* Input labels with better alignment and contrast */
    .stTextInput > label {
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.5rem;
        font-size: 1rem;
        display: block;
        width: 100%;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stSelectbox > label {
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.5rem;
        font-size: 1rem;
        display: block;
        width: 100%;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Column alignment fixes */
    .row-widget.stHorizontal {
        gap: 1rem;
        align-items: stretch;
    }
    
    .row-widget.stHorizontal > div {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: stretch;
    }
    
    /* Streamlit specific fixes */
    .stButton > button {
        width: 100%;
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Better spacing for containers */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Fix for expander alignment and contrast */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        color: #334155;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Ensure expander headers are visible */
    .streamlit-expanderHeader > div {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Target the expander header text specifically */
    .streamlit-expanderHeader span {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Ensure all expander elements are visible */
    .streamlit-expanderHeader * {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    /* Additional expander styling for better visibility */
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border-color: #0ea5e9;
    }
    
    /* Ensure expander content is also visible */
    .streamlit-expanderContent {
        color: #334155;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Ensure all text has good contrast */
    .stMarkdown {
        color: #334155;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        line-height: 1.6;
    }
    
    .stSubheader {
        color: #1e293b;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stHeader {
        color: #1e293b;
        font-weight: 700;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Better contrast for help text */
    .stMarkdown p {
        color: #475569;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        line-height: 1.6;
    }
    
    /* Ensure buttons have good text contrast */
    .stButton > button {
        color: white;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Fix for section headers to ensure they stand out */
    .stSubheader, .stHeader, h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important;
        font-weight: 600 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Ensure Streamlit subheaders are visible */
    .stSubheader {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
        margin-bottom: 1rem !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Make sure all headings are visible */
    h1, h2, h3 {
        color: #1e293b !important;
        font-weight: 700 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Citation and content text styling */
    .result-content {
        color: #334155;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 0.5rem 0;
    }
    
    .result-metadata {
        color: #64748b;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.25rem 0;
    }
    
    /* Link styling to ensure visibility */
    .result-metadata a {
        color: #0ea5e9 !important;
        text-decoration: underline;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .result-metadata a:hover {
        color: #0284c7 !important;
        text-decoration: none;
    }
    
    /* Ensure all links in results are visible */
    .results-container a {
        color: #0ea5e9 !important;
        text-decoration: underline;
        font-weight: 600;
    }
    
    .results-container a:hover {
        color: #0284c7 !important;
        text-decoration: none;
    }
    
    /* Error message styling */
    .error-message {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        border: 1px solid #ef4444;
        color: #991b1b;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 1rem;
        line-height: 1.5;
        font-weight: 500;
    }
    
    /* Success message styling */
    .success-message {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 1px solid #22c55e;
        color: #166534;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 1rem;
        line-height: 1.5;
        font-weight: 500;
    }
    
    /* Comprehensive error and warning styling for ALL Streamlit alerts */
    .stAlert {
        color: #334155 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        font-weight: 500 !important;
    }
    
    .stAlert * {
        color: #334155 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Error message styling */
    .stAlert[data-baseweb="notification"] {
        color: #991b1b !important;
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%) !important;
        border: 1px solid #ef4444 !important;
    }
    
    .stAlert[data-baseweb="notification"] * {
        color: #991b1b !important;
    }
    
    /* Warning message styling */
    .stAlert[data-baseweb="toast"] {
        color: #92400e !important;
        background: linear-gradient(135deg, #fffbeb 0%, #fed7aa 100%) !important;
        border: 1px solid #f59e0b !important;
    }
    
    .stAlert[data-baseweb="toast"] * {
        color: #92400e !important;
    }
    
    /* Success message styling */
    .stAlert[data-baseweb="banner"] {
        color: #166534 !important;
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%) !important;
        border: 1px solid #22c55e !important;
    }
    
    .stAlert[data-baseweb="banner"] * {
        color: #166534 !important;
    }
    
    /* Info message styling */
    .stAlert[data-baseweb="inline"] {
        color: #1e40af !important;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%) !important;
        border: 1px solid #3b82f6 !important;
    }
    
    .stAlert[data-baseweb="inline"] * {
        color: #1e40af !important;
    }
    
    /* Ensure all Streamlit elements are visible */
    .stText, .stText * {
        color: #334155 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Target specific error/warning elements */
    .stAlert, .stAlert *, .stAlert > div, .stAlert > div * {
        color: #334155 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        font-weight: 500 !important;
    }
    
    /* Ensure no white text anywhere */
    .stMarkdown *, .stText *, .stAlert *, .stSubheader *, .stHeader * {
        color: inherit !important;
    }
    
    /* Force visibility for all text elements */
    * {
        color: #334155 !important;
    }
    
    /* Override any white text */
    .stMarkdown, .stText, .stAlert, .stSubheader, .stHeader, p, span, div {
        color: #334155 !important;
    }
    
    /* Specific error styling for search errors and other notifications */
    .stAlert, .stAlert *, .stAlert > div, .stAlert > div *,
    .stAlert[data-baseweb="notification"], .stAlert[data-baseweb="notification"] *,
    .stAlert[data-baseweb="toast"], .stAlert[data-baseweb="toast"] *,
    .stAlert[data-baseweb="banner"], .stAlert[data-baseweb="banner"] *,
    .stAlert[data-baseweb="inline"], .stAlert[data-baseweb="inline"] * {
        color: #334155 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        font-weight: 500 !important;
    }
    
    /* Additional targeting for Streamlit error elements */
    .stAlert, .stAlert *, .stAlert > div, .stAlert > div *,
    .stAlert[data-baseweb="notification"], .stAlert[data-baseweb="notification"] *,
    .stAlert[data-baseweb="toast"], .stAlert[data-baseweb="toast"] *,
    .stAlert[data-baseweb="banner"], .stAlert[data-baseweb="banner"] *,
    .stAlert[data-baseweb="inline"], .stAlert[data-baseweb="inline"] *,
    .stAlert[data-baseweb="toast"], .stAlert[data-baseweb="toast"] *,
    .stAlert[data-baseweb="banner"], .stAlert[data-baseweb="banner"] *,
    .stAlert[data-baseweb="inline"], .stAlert[data-baseweb="inline"] * {
        color: #334155 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health() -> bool:
    """Check if the API is healthy."""
    try:
        response = requests.get("http://api:8000/healthz", timeout=5)
        return response.status_code == 200
    except:
        return False

def ingest_documents(link: str) -> Dict:
    """Ingest documents from Google Drive link."""
    try:
        # Extract folder ID from Google Drive link
        if "drive.google.com" in link:
            if "/folders/" in link:
                folder_id = link.split("/folders/")[1].split("?")[0]
            else:
                st.error("Please provide a valid Google Drive folder link")
                return {"status": "error"}
        else:
            st.error("Please provide a valid Google Drive folder link")
            return {"status": "error"}
        
        payload = {
            "source": "google_drive",
            "folder_id": folder_id
        }
        
        response = requests.post(
            "http://api:8000/ingest",
            json=payload,
            timeout=600  # 10 minutes timeout for ingestion
        )
        
        if response.status_code == 200:
            return response.json()
        else:
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
            "generate_answer": True
        }
        
        response = requests.post(
            "http://api:8000/query",
            json=payload,
            timeout=180
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search failed: {response.text}")
            return {"status": "error"}
            
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return {"status": "error"}

def main():
    """Main Luthro application."""
    
    # Header
    st.markdown('<h1 class="luthro-header">🔍 Luthro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="luthro-subtitle">Advanced AI-Powered Document Search & Analysis</p>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API service is not available. Please check if the backend is running.")
        st.stop()
    
    # Link Input Section
    # st.markdown('<div class="link-container">', unsafe_allow_html=True)
    st.subheader("📁 Document Ingestion")
    st.markdown("*Upload your documents to enable intelligent search and analysis*")
    
    col1, col2 = st.columns([4, 1], gap="small")
    
    with col1:
        link_input = st.text_input(
            "Google Drive Folder Link",
            placeholder="https://drive.google.com/drive/folders/...",
            help="Paste a Google Drive folder link to ingest documents for AI analysis"
        )
    
    with col2:
        # Add some spacing to align with the text input
        st.markdown("<div style='height: 3.5rem;'></div>", unsafe_allow_html=True)
        if st.button("🚀 Ingest Documents", key="ingest", use_container_width=True):
            if link_input:
                with st.spinner("Ingesting documents... This may take several minutes."):
                    result = ingest_documents(link_input)
                    if result.get("status") == "success":
                        st.markdown(f'<div class="status-success">✅ Successfully ingested {result.get("chunks_indexed", 0)} chunks from {result.get("documents_processed", 0)} documents!</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status-error">❌ Ingestion failed. Please check the link and try again.</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter a Google Drive folder link.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Search Section
    # st.markdown('<div class="search-container">', unsafe_allow_html=True)
    st.subheader("🔍 Intelligent Document Search")
    st.markdown("*Ask questions and get AI-powered answers from your documents*")
    
    # Search configuration
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="What are the main topics discussed in the documents?",
            help="Ask any question about your documents - the AI will find relevant information"
        )
    
    with col2:
        search_mode = st.selectbox(
            "Search Mode",
            options=[
                "elser_only",
                "dense_only", 
                "bm25_only",
                "dense_bm25",
                "full_hybrid"
            ],
            help="Choose your search strategy: ELSER (semantic), Dense (embeddings), BM25 (keywords), or Hybrid combinations"
        )
    
    with col3:
        top_k = st.selectbox(
            "Results Count",
            options=[3, 5, 10, 15, 20],
            index=1,
            help="Number of relevant results to return"
        )
    
    # Search button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Search Documents", key="search", use_container_width=True):
            if query:
                with st.spinner("Searching documents with AI..."):
                    results = search_documents(query, search_mode, top_k)
                    if results.get("status") == "success":
                        st.session_state.search_results = results
                        st.session_state.query = query
                        st.rerun()
                    else:
                        st.error("Search failed. Please try again.")
            else:
                st.warning("Please enter a search query.")
    
    # st.markdown('</div>', unsafe_allow_html=True)
    
    # Results Section
    if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        
        # Search summary
        results = st.session_state.search_results
        st.subheader(f"📊 Search Results for: '{st.session_state.query}'")
        
        # Search mode info with enhanced badge
        mode_descriptions = {
            "elser_only": "ELSER Semantic Search",
            "dense_only": "Dense Vector Search", 
            "bm25_only": "BM25 Keyword Search",
            "dense_bm25": "Dense + BM25 Hybrid",
            "full_hybrid": "ELSER + Dense + BM25 Full Hybrid"
        }
        
        st.markdown(f'<div class="mode-badge">🔧 {mode_descriptions.get(results.get("search_mode", ""), results.get("search_mode", ""))}</div>', unsafe_allow_html=True)
        
        # LLM Answer with enhanced styling
        if results.get('llm_response') and results['llm_response'].get('answer'):
            st.markdown("### 💡 AI Generated Answer")
            st.markdown(f'<div class="ai-answer">{results["llm_response"]["answer"]}</div>', unsafe_allow_html=True)
        
        # Search Results with enhanced cards
        if results.get('results'):
            st.markdown(f"### 📚 Found {len(results['results'])} Results")
            
            for i, result in enumerate(results['results']):
                with st.expander(f"Result {i+1} - {result.get('filename', 'Unknown')} (Score: {result.get('_score', 0):.3f})"):
                    # Content with proper styling
                    st.markdown(f'<div class="result-content"><strong>Content:</strong> {result.get("content", "No content")}</div>', unsafe_allow_html=True)
                    
                    # Metadata in columns with proper styling
                    col1, col2, col3 = st.columns(3)
                with col1:
                        st.markdown(f'<div class="result-metadata"><strong>File:</strong> {result.get("filename", "Unknown")}</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="result-metadata"><strong>Chunk ID:</strong> {result.get("chunk_id", "N/A")}</div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="result-metadata"><strong>Search Type:</strong> {result.get("search_type", "Unknown")}</div>', unsafe_allow_html=True)
                
                # Source link with proper styling
                if result.get('file_url'):
                        st.markdown(f'<div class="result-metadata"><strong>Source:</strong> <a href="{result.get("file_url", "")}" target="_blank">{result.get("file_url", "")}</a></div>', unsafe_allow_html=True)
        
        # Clear results button
        if st.button("🗑️ Clear Results", key="clear"):
            if 'search_results' in st.session_state:
                del st.session_state.search_results
            if 'query' in st.session_state:
                del st.session_state.query
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
