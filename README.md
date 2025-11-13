# AI-Powered Knowledge Base (RAG Pipeline)

A complete RAG (Retrieval-Augmented Generation) system with FastAPI backend and Streamlit frontend for document ingestion, semantic search, and AI-powered Q&A.

## 🚀 Quick Start

```bash
./start.sh
```

Then open http://localhost:8501 in your browser.

## ✨ Features

- **Document Ingestion**: Add text documents with automatic chunking and embedding
- **Semantic Search**: Find relevant information using natural language queries  
- **Question Answering**: Get AI-generated answers based on your knowledge base
- **Completeness Analysis**: Analyze coverage gaps in your documents
- **Incremental Indexing**: Only re-processes changed documents (hash-based detection)

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  ┌──────────────┐              ┌──────────────┐            │
│  │  Streamlit   │              │  API Clients │            │
│  │   Frontend   │              │  (REST/HTTP) │            │
│  └──────┬───────┘              └──────┬───────┘            │
└─────────┼──────────────────────────────┼────────────────────┘
          │                              │
          └──────────────┬───────────────┘
                         │
          ┌──────────────▼───────────────┐
          │      FastAPI Backend         │
          │    (REST API Endpoints)      │
          └──────────────┬───────────────┘
                         │
          ┌──────────────┴───────────────┐
          │                              │
    ┌─────▼──────┐              ┌───────▼────────┐
    │ Ingestion  │              │   RAG          │
    │  Service   │              │  Retriever     │
    └─────┬──────┘              └───────┬────────┘
          │                              │
          │                              │
    ┌─────┴──────┐              ┌───────┴────────┐
    │            │              │                │
┌───▼────┐  ┌───▼────┐    ┌────▼────┐    ┌─────▼─────┐
│ Qdrant │  │Postgres│    │Sentence │    │   LLM     │
│(Vector │  │(Meta)  │    │Transform│    │ (Gemini/  │
│  DB)   │  │        │    │  Model  │    │  OpenAI)  │
└────────┘  └────────┘    └─────────┘    └───────────┘
```

### Data Flow

#### Document Ingestion Flow
```
1. Document Upload
   ↓
2. Content Hash Calculation (SHA-256)
   ↓
3. Check PostgreSQL for existing document
   ├─→ If unchanged: Skip (return "skipped")
   └─→ If changed/new: Continue
   ↓
4. Delete old vectors from Qdrant (if updating)
   ↓
5. Text Chunking (1024 chars, 200 overlap)
   ↓
6. Generate Embeddings (all-MiniLM-L6-v2)
   ↓
7. Store in Qdrant (vectors + metadata)
   ↓
8. Update PostgreSQL (hash + timestamp)
```

#### Query Flow (RAG)
```
1. User Question
   ↓
2. Generate Query Embedding
   ↓
3. Vector Search in Qdrant (cosine similarity)
   ↓
4. Retrieve Top-K Relevant Chunks
   ↓
5. Construct Prompt (Context + Question)
   ↓
6. LLM Generation (Gemini/OpenAI)
   ↓
7. Return Answer + Source Citations
```

## 🛠️ Tech Stack

- **FastAPI**: Async REST API backend with automatic OpenAPI docs
- **Streamlit**: Interactive web frontend for easy testing
- **Qdrant**: Vector database for efficient similarity search
- **PostgreSQL**: Relational database for document metadata
- **Sentence-Transformers**: Local embedding generation (all-MiniLM-L6-v2)
- **Google Gemini / OpenAI**: LLM for answer generation
- **Docker Compose**: Service orchestration for Qdrant and PostgreSQL

## 📦 Installation

### Prerequisites
- Python 3.8+
- Docker and Docker Compose
- API key for Gemini or OpenAI

### Setup

1. **Clone and Install Dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure Environment**
Create a `.env` file:
```env
# LLM Configuration
LLM_PROVIDER=gemini  # or "openai"
GEMINI_API_KEY=your_gemini_api_key_here
# OR
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gemini-1.5-flash  # or "gpt-3.5-turbo"

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=knowledge_base

# PostgreSQL Configuration
POSTGRES_DSN=postgresql://rag_user:rag_password@localhost:5433/rag_database

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
```

3. **Start Services**
```bash
# Start Docker services (Qdrant + PostgreSQL)
docker-compose up -d

# Start backend API
python main.py

# In another terminal, start frontend
streamlit run frontend.py
```

## 📡 API Endpoints

### Document Management
- `POST /ingest/text` - Ingest text document
  ```json
  {
    "file_path": "docs/intro.txt",
    "content": "Your document content here..."
  }
  ```

- `POST /ingest/file` - Upload and ingest file (multipart/form-data)
- `DELETE /documents/{file_path}` - Delete a document

### Search & Query
- `POST /search` - Semantic search
  ```json
  {
    "query": "machine learning algorithms",
    "top_k": 5
  }
  ```

- `POST /query/qa` - Question answering
  ```json
  {
    "question": "What is machine learning?",
    "top_k": 5
  }
  ```

- `POST /query/completeness` - Coverage analysis
  ```json
  {
    "topic": "neural networks",
    "top_k": 10
  }
  ```

### System
- `GET /health` - Health check
- `GET /stats` - Collection statistics
- `GET /` - API information
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_ingestion.py

# Run with verbose output
pytest -v
```

### Test Structure
```
tests/
├── test_ingestion.py      # Unit tests for ingestion service
├── test_retriever.py      # Unit tests for RAG retriever
├── test_database.py       # Unit tests for database operations
└── test_api.py            # Integration tests for API endpoints
```

## 🎓 Design Decisions

### 1. Incremental Indexing Strategy
**Decision**: Hash-based change detection using SHA-256  
**Rationale**: 
- Prevents unnecessary re-indexing of unchanged documents
- Reduces computational cost and storage overhead
- Enables efficient updates for large document collections
- Hash comparison is O(1) vs. full content comparison O(n)

**Implementation**: Content hash stored in PostgreSQL, compared before indexing

### 2. Chunking Strategy
**Decision**: 1024 characters with 200 character overlap  
**Rationale**:
- **1024 chars**: Balances context preservation with embedding quality
  - Too small: Loses context, more chunks to manage
  - Too large: Embedding quality degrades, less precise retrieval
- **200 overlap**: Ensures continuity across chunk boundaries
  - Prevents information loss at chunk edges
  - Maintains semantic coherence

**Trade-off**: Slightly increased storage (~20% overhead) for better retrieval quality

### 3. Dual Storage Architecture
**Decision**: Qdrant for vectors, PostgreSQL for metadata  
**Rationale**:
- **Separation of concerns**: Each database optimized for its purpose
- **Qdrant**: Specialized for vector similarity search (HNSW algorithm)
- **PostgreSQL**: Relational queries for metadata, change tracking, analytics
- **Scalability**: Can scale vector DB and metadata DB independently

**Alternative considered**: Single database (e.g., pgvector) - rejected due to performance limitations

### 4. Embedding Model Choice
**Decision**: `all-MiniLM-L6-v2` (Sentence Transformers)  
**Rationale**:
- **384 dimensions**: Good balance between quality and speed
- **Fast inference**: Local model, no API calls needed
- **Good quality**: Competitive with larger models for semantic similarity
- **Multilingual support**: Works across languages
- **Small size**: ~80MB, easy to deploy

**Trade-off**: Slightly lower quality than larger models (e.g., OpenAI embeddings) but much faster and free

### 5. RAG Architecture
**Decision**: Retrieval-Augmented Generation with top-k context  
**Rationale**:
- **Accuracy**: Answers grounded in actual documents
- **Source attribution**: Can cite sources for verification
- **Reduced hallucinations**: LLM constrained to retrieved context
- **Efficiency**: Only relevant context sent to LLM (cost-effective)

**Implementation**: Semantic search → Top-K chunks → Prompt construction → LLM generation

### 6. Modular Service Design
**Decision**: Separate services (Ingestion, Retrieval, Database)  
**Rationale**:
- **Maintainability**: Clear separation of concerns
- **Testability**: Each service can be tested independently
- **Extensibility**: Easy to swap implementations (e.g., different embedding models)
- **Reusability**: Services can be used in different contexts

## ⚖️ Trade-offs

See [TRADE_OFFS.md](TRADE_OFFS.md) for detailed trade-off analysis covering:
- Chunking size and overlap
- Embedding model selection
- Vector database choice (Qdrant vs alternatives)
- Metadata storage (PostgreSQL vs alternatives)
- LLM provider selection
- And more...

## 🚀 Performance Considerations

### Current Performance Characteristics

**Ingestion**:
- Small documents (<10KB): ~100-200ms
- Medium documents (10-100KB): ~500ms-2s
- Large documents (>100KB): ~2-10s (depends on chunk count)

**Search**:
- Query latency: ~50-150ms (includes embedding generation + vector search)
- Scales well: O(log n) with HNSW indexing in Qdrant

**Q&A**:
- Latency: ~1-5s (depends on LLM provider and response length)
- Bottleneck: LLM API call (network latency)

### Optimization Strategies

1. **Batch Processing**: For bulk ingestion, process multiple documents in parallel
2. **Caching**: Cache frequent queries (not implemented, but recommended)
3. **Async Processing**: Large file ingestion could be moved to background tasks
4. **Embedding Caching**: Cache embeddings for repeated queries
5. **Connection Pooling**: Already implemented for PostgreSQL

### Scalability

- **Vector Search**: Qdrant handles millions of vectors efficiently
- **Metadata**: PostgreSQL can scale with proper indexing
- **API**: FastAPI supports async operations for concurrent requests
- **Bottleneck**: LLM API rate limits (consider queuing for high volume)

## 📁 Project Structure

```
wandai-task2/
├── main.py                 # FastAPI application & endpoints
├── ingestion.py            # Document processing & indexing
├── retriever.py            # Semantic search & RAG
├── database.py             # PostgreSQL operations
├── frontend.py             # Streamlit UI
├── docker-compose.yml      # Qdrant + PostgreSQL services
├── requirements.txt        # Python dependencies
├── start.sh               # Startup script
├── .env                   # Configuration (create from template)
├── tests/                 # Test suite
│   ├── test_ingestion.py
│   ├── test_retriever.py
│   ├── test_database.py
│   └── test_api.py
├── qdrant_storage/        # Vector database storage (auto-created)
├── logs/                  # Application logs
├── README.md              # This file
└── TRADE_OFFS.md          # Detailed trade-off analysis
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider: "gemini" or "openai" | "openai" |
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `LLM_MODEL` | Model name | "gpt-3.5-turbo" |
| `QDRANT_HOST` | Qdrant host | "localhost" |
| `QDRANT_PORT` | Qdrant port | 6333 |
| `QDRANT_COLLECTION_NAME` | Collection name | "knowledge_base" |
| `POSTGRES_DSN` | PostgreSQL connection string | - |
| `APP_HOST` | API host | "0.0.0.0" |
| `APP_PORT` | API port | 8000 |

## 🐛 Troubleshooting

### Common Issues

1. **Qdrant connection error**
   - Ensure Docker is running: `docker ps`
   - Check Qdrant is up: `docker-compose ps`
   - Verify port 6333 is available

2. **PostgreSQL connection error**
   - Check PostgreSQL is running: `docker-compose ps`
   - Verify connection string in `.env`
   - Ensure database exists (auto-created on first run)

3. **LLM API errors**
   - Verify API key is set correctly in `.env`
   - Check API key is valid and has credits
   - For Gemini: Ensure model name is correct (e.g., "gemini-1.5-flash")

4. **Import errors**
   - Ensure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

## 📝 License

MIT

## 🙏 Acknowledgments

- Qdrant for the excellent vector database
- Sentence Transformers for embedding models
- FastAPI and Streamlit communities
