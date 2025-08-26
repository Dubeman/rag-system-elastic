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
| **Search Engine** | Elasticsearch | ELSER sparse embeddings + dense + BM25 |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Fast, efficient, free |
| **LLM** | Llama-3/Mistral (HuggingFace/Ollama) | Open source, customizable |
| **Backend** | FastAPI | Standard for ML services |
| **Frontend** | Streamlit | Rapid prototyping |
| **Infrastructure** | Docker + Docker Compose | Reproducibility |
| **CI/CD** | GitHub Actions | Automated testing and deployment |

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-system-elastic
   ```

2. **Environment setup**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start services**
   ```bash
   docker-compose up -d
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Initialize the system**
   ```bash
   python scripts/setup.py
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
pytest tests/ -v --cov=src/
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
- [ ] Ingestion pipeline
- [ ] Elasticsearch indexing
- [ ] Hybrid retrieval
- [ ] LLM integration

### 📋 Phase 3: API & UI
- [ ] FastAPI service
- [ ] Streamlit interface
- [ ] End-to-end testing

### 🚀 Phase 4: Production Readiness
- [ ] Performance optimization
- [ ] Monitoring & logging
- [ ] Documentation completion

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
