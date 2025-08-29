# 🔍 RAG System with Elasticsearch

A modern, production-ready Retrieval-Augmented Generation (RAG) system built with Elasticsearch, open-source LLMs, and industry best practices.

## 📋 Overview

This project implements a sophisticated RAG pipeline that combines multiple retrieval strategies (ELSER sparse embeddings, dense embeddings, and BM25) with open-source language models to provide accurate, context-aware answers with proper citations.

### 🎯 Key Features

- **Hybrid Retrieval**: ELSER + Dense Embeddings + BM25 with Reciprocal Rank Fusion (RRF)
- **Open Source LLMs**: Integration with Llama-3/Mistral via HuggingFace/Ollama
- **Google Drive Integration**: Direct PDF ingestion from Google Drive
- **Production Ready**: Docker containerization, CI/CD, comprehensive testing
- **Modern Architecture**: Modular design following ML engineering best practices

## 🏗️ System Architecture

![System Architecture](Hexaware%20internship.drawio.png)

The system follows a modular microservices architecture:

```
Google Drive PDFs → Ingestion → Elasticsearch → Retrieval → FastAPI → LLM → Answer + Citations → Streamlit UI
                                    ↓
                            [ELSER + Dense + BM25]
                                    ↓
                              [RRF Fusion]
```

### Components

- **Ingestion Layer**: PDF processing and chunking from Google Drive
- **Indexing Layer**: Multi-modal indexing with ELSER, dense embeddings, and BM25
- **Retrieval Layer**: Hybrid search with configurable fusion strategies
- **Generation Layer**: Open-source LLM integration with guardrails
- **API Layer**: FastAPI service with request validation
- **UI Layer**: Clean Streamlit interface for demonstrations

## 🛠️ Tech Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Search Engine** | Elasticsearch 8.11.0 | ELSER sparse embeddings + dense + BM25 |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Fast, efficient, free |
| **LLM** | Llama-3/Mistral (HuggingFace/Ollama) | Open source, customizable |
| **Backend** | FastAPI | Standard for ML services |
| **Frontend** | Streamlit | Rapid prototyping |
| **Infrastructure** | Docker + Docker Compose | Reproducibility |
| **CI/CD** | GitHub Actions | Automated testing and deployment |

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose** (v2.0+)
- **At least 8GB RAM** (Elasticsearch + Ollama)
- **Google Drive API credentials** (for PDF ingestion)

### Docker Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-system-elastic
   ```

2. **Environment setup (optional)**
   ```bash
   cp env.example .env
   # Edit .env if you need custom settings (most have sensible defaults)
   ```

3. **Build and start all services**
   ```bash
   docker compose up --build -d
   ```

4. **Monitor service startup**
   ```bash
   docker compose logs -f
   ```

5. **Wait for services to be ready**
   - Elasticsearch: ~2-3 minutes (includes ELSER model download)
   - Ollama: ~5-10 minutes (includes model download)
   - API & UI: ~1 minute

6. **Access the system**
   - **API**: http://localhost:8000
   - **UI**: http://localhost:8501
   - **Elasticsearch**: http://localhost:9200
   - **Ollama**: http://localhost:11434

### 🖥️ User Interface

![UI POC](ui%20poc.png)

The system provides a clean Streamlit interface for:
- **Document Ingestion**: Upload and process PDFs from Google Drive
- **Query Interface**: Ask questions and get AI-generated answers with citations
- **Search Results**: View retrieved documents and their relevance scores
- **Response Generation**: Get detailed answers with proper source attribution

## 🔧 Docker Commands

### Essential Commands
```bash
# Build and start all services
docker compose up --build -d

# Start services (if already built)
docker compose up -d

# Stop all services
docker compose down

# View logs
docker compose logs -f

# Restart specific service
docker compose restart api
docker compose restart ui

# Check service status
docker compose ps

# Rebuild specific service
docker compose build api
docker compose build ui
```

### Service Management
```bash
# Start specific services
docker compose up -d elasticsearch
docker compose up -d api
docker compose up -d ui
docker compose up -d ollama

# View logs for specific service
docker compose logs elasticsearch
docker compose logs api
docker compose logs ui
docker compose logs ollama

# Execute commands inside containers
docker compose exec api bash
docker compose exec elasticsearch bash
```

## ⚙️ Configuration

### Environment Variables

Copy `env.example` to `.env` and configure:

```bash
# Elasticsearch Configuration
ELASTICSEARCH_URL=http://localhost:9200

# LLM Configuration
LLM_SERVICE_URL=http://ollama:11434
LLM_MODEL_NAME=tinyllama
LLM_TIMEOUT=180
```

**Note**: Most configuration is handled automatically with sensible defaults. The system works out-of-the-box with minimal setup.

### Google Drive Setup

The system automatically uses public Google Drive access, so no credentials are required. Simply:

1. **Upload PDFs to a Google Drive folder**
2. **Make the folder public** (or share with "Anyone with the link can view")
3. **Use the folder ID in your API requests**

**Example usage:**
```bash
# Ingest documents from a public Google Drive folder
curl -X POST "http://localhost:8000/ingest" \
     -H "Content-Type: application/json" \
     -d '{"source": "google_drive", "folder_id": "your-folder-id"}'
```



## 📁 Project Structure

```
rag-system-elastic/
├── src/
│   ├── ingestion/          # PDF processing and chunking
│   ├── indexing/           # Elasticsearch indexing logic
│   ├── retrieval/          # Hybrid search implementation
│   ├── generation/         # LLM integration and response generation
│   ├── api/                # FastAPI service
│   └── ui/                 # Streamlit interface
├── tests/                  # Unit and integration tests
├── docker/                 # Docker configurations
├── scripts/                # Setup and utility scripts
├── docs/                   # Documentation
├── data/                   # Data storage
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Service orchestration
├── .pre-commit-config.yaml # Code quality hooks
└── README.md              # This file
```

## 🔧 Development

### Code Quality

We use pre-commit hooks for consistent code quality:

```bash
pip install pre-commit
pre-commit install
```

### Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v --cov=src/

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v

# Run with coverage report
pytest tests/ --cov=src/ --cov-report=html
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/your-feature`
2. Write tests first (TDD approach)
3. Implement the feature
4. Ensure all tests pass
5. Submit pull request

## 📡 API Endpoints

### Core Endpoints

- `POST /query` - Submit questions and get answers with citations
- `POST /ingest` - Trigger document ingestion and reindexing
- `GET /healthz` - Health check endpoint

### Example Usage

```bash
# Query the system
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is machine learning?", "top_k": 5}'

# Ingest new documents
curl -X POST "http://localhost:8000/ingest" \
     -H "Content-Type: application/json" \
     -d '{"source": "google_drive", "folder_id": "your-folder-id"}'

# Health check
curl "http://localhost:8000/healthz"
```

### 🔧 Testing Commands

For comprehensive testing commands, see: **[docker_curl_commands.txt](docker_curl_commands.txt)**

Key working test examples:
```bash
# Test with BM25 only
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is a Dockerfile and how does it work?",
    "search_mode": "bm25_only",
    "top_k": 5,
    "generate_answer": true
  }'

# Test with full hybrid search
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are CSRF tokens in the FortiOS REST API reference?",
    "search_mode": "full_hybrid",
    "top_k": 5,
    "generate_answer": true
  }'
```

## 🎛️ Configuration

### Retrieval Modes

- **ELSER Only**: Pure sparse retrieval
- **Hybrid**: ELSER + Dense + BM25 with RRF fusion (recommended)
- **Dense Only**: Vector similarity search
- **BM25 Only**: Traditional keyword search

### Performance Tuning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 300 | Token chunk size for documents (hardcoded) |
| `top_k` | 5 | Number of documents to retrieve (configurable via API) |
| `rrf_rank_constant` | 60 | RRF fusion parameter (hardcoded) |
| `temperature` | 0.1 | LLM generation temperature (hardcoded) |

**Note**: Most parameters have sensible defaults and are hardcoded for optimal performance. Customization is available through API parameters when needed.

## 🛡️ Guardrails & Safety

- **Content Filtering**: Harmful/off-topic query detection
- **Uncertainty Handling**: "I don't know" responses for low-confidence answers
- **Citation Integrity**: All answers include source references
- **Rate Limiting**: API endpoint protection

## 📊 Performance Targets

- **Latency**: ≤ 3 seconds end-to-end : Curent ~ 1 minute


## 🐛 Troubleshooting

### Common Issues

1. **Elasticsearch won't start**
   ```bash
   # Check system resources
   docker system df
   docker stats
   
   # Increase Docker memory limit to 8GB+
   # Restart Docker Desktop
   ```

2. **ELSER model download issues**
   ```bash
   # Check Elasticsearch logs
   docker compose logs elasticsearch
   
   # Verify internet connection
   # Check available disk space
   ```

3. **Ollama model issues**
   ```bash
   # Check Ollama logs
   docker compose logs ollama
   
   # Check available models
   curl http://localhost:11434/api/tags
   
   # Pull model manually
   curl -X POST "http://localhost:11434/api/pull" \
        -H "Content-Type: application/json" \
        -d '{"name": "tinyllama"}'
   ```

4. **API connection errors**
   ```bash
   # Check service health
   docker compose ps
   
   # Verify network connectivity
   docker network ls
   docker network inspect rag-system-elastic_rag-network
   ```

### Service Health Checks

```bash
# Check all services
docker compose ps

# Check specific service logs
docker compose logs elasticsearch
docker compose logs api
docker compose logs ui
docker compose logs ollama

# Restart specific service
docker compose restart api

# Full system restart
docker compose down
docker compose up --build -d
```


