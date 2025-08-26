"""Streamlit UI for RAG System."""

import streamlit as st
import requests
import json
import time
import os
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")

def check_api_health() -> bool:
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/healthz", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

def ingest_documents(source: str, folder_id: str = "", sample_text: str = "") -> Dict:
    """Call the ingestion API."""
    try:
        payload = {
            "source": source,
            "folder_id": folder_id,
            "sample_text": sample_text
        }
        
        response = requests.post(
            f"{API_BASE_URL}/ingest",
            json=payload,
            timeout=300  # 5 minutes timeout for ingestion
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ingestion failed: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error calling ingestion API: {e}")
        return None

def query_documents(question: str, top_k: int = 5) -> Dict:
    """Call the query API."""
    try:
        payload = {
            "question": question,
            "top_k": top_k
        }
        
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Query failed: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error calling query API: {e}")
        return None

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🔍 RAG System Dashboard")
    st.markdown("*Retrieval-Augmented Generation with Elasticsearch*")
    
    # Check API health
    if check_api_health():
        st.success("✅ API is healthy and ready")
    else:
        st.error("❌ API is not accessible. Please ensure the FastAPI server is running.")
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    tab = st.sidebar.radio(
        "Choose Action:",
        ["📤 Ingest Documents", "🔍 Search Documents", "📊 System Status"]
    )
    
    if tab == "📤 Ingest Documents":
        ingest_tab()
    elif tab == "🔍 Search Documents":
        search_tab()
    elif tab == "📊 System Status":
        status_tab()

def ingest_tab():
    """Document ingestion interface."""
    st.header("📤 Document Ingestion")
    
    # Source selection
    source_type = st.selectbox(
        "Select Document Source:",
        ["sample", "google_drive"],
        help="Choose whether to ingest sample text or documents from Google Drive"
    )
    
    if source_type == "sample":
        st.subheader("Sample Text Ingestion")
        
        sample_text = st.text_area(
            "Enter sample text:",
            value="This is a sample document for testing the RAG system. It contains some text that will be chunked and indexed in Elasticsearch.",
            height=150,
            help="Enter any text you want to index for testing"
        )
        
        if st.button("🚀 Ingest Sample Text", type="primary"):
            if sample_text.strip():
                with st.spinner("Processing sample text..."):
                    result = ingest_documents("sample", sample_text=sample_text)
                    
                if result:
                    display_ingestion_result(result)
            else:
                st.warning("Please enter some sample text")
    
    elif source_type == "google_drive":
        st.subheader("Google Drive Ingestion")
        
        st.info("📝 **Instructions:** Paste the Google Drive folder URL below. The system will automatically extract the folder ID and download PDFs.")
        
        # Google Drive URL input
        drive_url = st.text_input(
            "Google Drive Folder URL:",
            placeholder="https://drive.google.com/drive/folders/1h6GptTW3DPCdhu7q5tY-83CXrpV8TmY_",
            help="Paste the full Google Drive folder URL here"
        )
        
        # Extract folder ID from URL
        folder_id = ""
        if drive_url:
            if "folders/" in drive_url:
                try:
                    folder_id = drive_url.split("folders/")[1].split("?")[0].split("/")[0]
                    st.success(f"✅ Folder ID extracted: `{folder_id}`")
                except:
                    st.error("❌ Could not extract folder ID from URL")
            else:
                st.error("❌ Invalid Google Drive folder URL format")
        
        if st.button("🚀 Ingest from Google Drive", type="primary", disabled=not folder_id):
            if folder_id:
                with st.spinner("Fetching and processing documents from Google Drive..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate progress updates (since we can't get real-time progress from the API)
                    for i in range(10):
                        progress_bar.progress((i + 1) * 10)
                        status_text.text(f"Processing... Step {i + 1}/10")
                        time.sleep(0.5)
                    
                    result = ingest_documents("google_drive", folder_id=folder_id)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    
                if result:
                    display_ingestion_result(result)
            else:
                st.warning("Please enter a valid Google Drive folder URL")

def display_ingestion_result(result: Dict):
    """Display ingestion results."""
    st.success("✅ Ingestion completed successfully!")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents Processed", result.get("documents_processed", 0))
    
    with col2:
        st.metric("Chunks Indexed", result.get("chunks_indexed", 0))
    
    with col3:
        st.metric("Errors", result.get("errors", 0))
    
    with col4:
        st.metric("Source", result.get("source", "unknown"))
    
    # Detailed result
    with st.expander("📋 Detailed Results"):
        st.json(result)

def search_tab():
    """Document search interface."""
    st.header("🔍 Search Documents")
    
    # Search configuration
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_input(
            "Enter your question:",
            placeholder="What is the main topic discussed in the documents?",
            help="Ask any question about the indexed documents"
        )
    
    with col2:
        top_k = st.selectbox("Results to return:", [3, 5, 10], index=1)
    
    # Search button
    if st.button("🔍 Search", type="primary", disabled=not question.strip()):
        if question.strip():
            with st.spinner("Searching documents..."):
                results = query_documents(question, top_k)
            
            if results:
                display_search_results(results)
        else:
            st.warning("Please enter a question")
    
    # Search tips
    with st.expander("💡 Search Tips"):
        st.markdown("""
        - Ask specific questions about the content
        - Use natural language queries
        - Try different phrasings if you don't get relevant results
        - The system searches through chunked document content
        """)

def display_search_results(results: Dict):
    """Display search results."""
    st.subheader(f"🎯 Search Results for: *{results.get('question', '')}*")
    
    search_results = results.get("results", [])
    total_results = results.get("total_results", 0)
    
    if total_results == 0:
        st.warning("No results found. Try rephrasing your question or check if documents are indexed.")
        return
    
    st.info(f"Found {total_results} relevant chunks")
    
    # Display results
    for i, result in enumerate(search_results, 1):
        with st.expander(f"📄 Result {i} - Score: {result.get('_score', 'N/A'):.3f}"):
            source = result.get("_source", {})
            
            # Display metadata
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Document:** {source.get('filename', 'Unknown')}")
                st.write(f"**Chunk ID:** {source.get('chunk_id', 'N/A')}")
            
            with col2:
                st.write(f"**File URL:** {source.get('file_url', 'N/A')}")
                st.write(f"**Modified:** {source.get('modified_time', 'N/A')}")
            
            # Display content
            st.markdown("**Content:**")
            content = source.get('content', 'No content available')
            st.markdown(f"```\n{content}\n```")

def status_tab():
    """System status interface."""
    st.header("📊 System Status")
    
    # API Health Check
    if st.button("🔄 Refresh Status"):
        st.rerun()
    
    # Health status
    if check_api_health():
        st.success("✅ FastAPI Server: Healthy")
        
        # Try to get more detailed health info
        try:
            response = requests.get(f"{API_BASE_URL}/healthz")
            health_data = response.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("API Status", health_data.get("status", "unknown"))
                st.metric("Elasticsearch", health_data.get("elasticsearch", "unknown"))
            
            with col2:
                services = health_data.get("services", [])
                st.write("**Active Services:**")
                for service in services:
                    st.write(f"• {service}")
            
            with st.expander("📋 Detailed Health Info"):
                st.json(health_data)
                
        except Exception as e:
            st.warning(f"Could not get detailed health info: {e}")
    else:
        st.error("❌ FastAPI Server: Not accessible")
        
        st.markdown("""
        **Troubleshooting:**
        - Ensure Docker containers are running: `docker compose up`
        - Check if FastAPI is running on port 8000
        - Verify Elasticsearch is accessible
        """)
    
    # System Information
    st.subheader("🛠️ System Information")
    
    info_data = {
        "API Base URL": API_BASE_URL,
        "Expected Endpoints": ["/healthz", "/ingest", "/query"],
        "Supported Sources": ["sample", "google_drive"],
        "Frontend": "Streamlit",
        "Backend": "FastAPI + Elasticsearch"
    }
    
    for key, value in info_data.items():
        if isinstance(value, list):
            st.write(f"**{key}:** {', '.join(value)}")
        else:
            st.write(f"**{key}:** {value}")

if __name__ == "__main__":
    main()
