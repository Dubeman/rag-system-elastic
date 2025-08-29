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

2. **Environment setup**
   ```bash
   cp env.example .env
   # Edit .env with your Google Drive credentials and folder ID
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
ELASTICSEARCH_INDEX=rag_documents

# Google Drive API Configuration
GOOGLE_DRIVE_CREDENTIALS_PATH=./credentials/google_drive_credentials.json
GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here

# LLM Configuration
LLM_MODEL_NAME=microsoft/DialoGPT-medium
LLM_MAX_LENGTH=512
LLM_TEMPERATURE=0.1
HF_HOME=./models

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# UI Configuration
STREAMLIT_HOST=0.0.0.0
STREAMLIT_PORT=8501

# Retrieval Configuration
DEFAULT_TOP_K=5
RRF_RANK_CONSTANT=60
CHUNK_SIZE=300
CHUNK_OVERLAP=50

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Environment
ENVIRONMENT=development
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
| `chunk_size` | 300 | Token chunk size for documents |
| `top_k` | 5 | Number of documents to retrieve |
| `rrf_rank_constant` | 60 | RRF fusion parameter |
| `temperature` | 0.1 | LLM generation temperature |

## 🛡️ Guardrails & Safety

- **Content Filtering**: Harmful/off-topic query detection
- **Uncertainty Handling**: "I don't know" responses for low-confidence answers
- **Citation Integrity**: All answers include source references
- **Rate Limiting**: API endpoint protection

## 📊 Performance Targets

- **Latency**: ≤ 3 seconds end-to-end
- **Accuracy**: Measured via human evaluation
- **Availability**: 99.9% uptime
- **Scalability**: Handles 100+ concurrent users

## 🔄 Development Phases

### ✅ Phase 1: Foundation & Design
- [x] Architecture design
- [x] Tech stack selection
- [x] Project structure setup

### 🚧 Phase 2: Core Implementation
- [x] Ingestion pipeline
- [x] Elasticsearch indexing
- [x] Hybrid retrieval
- [x] LLM integration

### 📋 Phase 3: API & UI
- [x] FastAPI service
- [x] Streamlit interface
- [x] End-to-end testing

### 🚀 Phase 4: Production Readiness
- [x] Performance optimization
- [x] Monitoring & logging
- [x] Documentation completion

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

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Elasticsearch team for ELSER technology
- HuggingFace for open-source transformer models
- The open-source ML community

## 📞 Support

For questions or support:
- Create an issue on GitHub
- Check the [documentation](docs/)
- Review the [FAQ](docs/FAQ.md)

---

*Built with ❤️ using modern AI/ML engineering practices*
