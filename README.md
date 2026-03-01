<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-00D4A8?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Next.js-14-black?logo=next.js&logoColor=white" alt="Next.js" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Qdrant-Vector_DB-DC382D?logo=data:image/svg+xml;base64,&logoColor=white" alt="Qdrant" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License" />
</p>

# 🧠 Smart Notes Assistant

> **An AI-powered knowledge assistant that lets you upload documents and ask questions — powered by a production-grade RAG pipeline with hybrid search, cross-encoder reranking, and streaming chat.**

Upload your PDFs, text files, or markdown docs. Ask questions in natural language. Get accurate, cited answers with source references — all running locally on your machine.

---

## ✨ Features

- 🔍 **Hybrid Retrieval** — Combines dense vector search (semantic) + BM25 (keyword) for best-of-both-worlds document retrieval
- 🏆 **Cross-Encoder Reranking** — MS MARCO model re-scores candidates for precision
- 📄 **Multi-Format Upload** — Supports PDF, TXT, Markdown, CSV, and JSON documents
- 💬 **Streaming Chat** — Real-time token-by-token responses via Server-Sent Events
- 📚 **Source Citations** — Every answer includes clickable source references with relevance scores
- 🔒 **Fully Local** — Runs entirely on your machine with Ollama (or optionally use OpenAI)
- 🎨 **Modern UI** — ChatGPT-inspired dark interface with conversation history and drag-and-drop upload
- 🐳 **One-Command Setup** — Single `docker compose up` to launch everything

---

## 🖥️ Screenshots

<details>
<summary>Click to expand</summary>

### Welcome Screen
The landing page with suggestion cards to get you started.

### Chat Interface
Ask questions and get answers with source citations.

### Document Upload
Drag-and-drop your files or browse to upload.

</details>

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Browser (localhost:3000)                         │
│                   Next.js 14 · React 18 · Tailwind CSS                  │
│              Streaming Chat · Document Upload · Dark Theme               │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │ REST + SSE
┌───────────────────────────────▼──────────────────────────────────────────┐
│                       FastAPI Backend (:8000)                            │
│                                                                          │
│   ┌───────────┐    ┌─────────────────────────────────────────────────┐   │
│   │  Endpoints │    │            RETRIEVAL PIPELINE                   │   │
│   │            │    │                                                 │   │
│   │  /api/chat │───▶│  1. Query Rewriting ─── LLM expands query      │   │
│   │  /api/     │    │  2. Dense Retrieval ─── Qdrant ANN (top 20)    │   │
│   │   ingest   │    │  3. BM25 Retrieval ──── Keyword match (top 20) │   │
│   │  /api/     │    │  4. Score Fusion ────── Reciprocal Rank Fusion │   │
│   │   admin    │    │  5. Cross-Encoder ───── Rerank to top 5        │   │
│   │  /health   │    │  6. Compression ─────── Extract relevant parts │   │
│   └───────────┘    │  7. Prompt Template ─── Inject context + query  │   │
│                     └─────────────────────────────────────────────────┘   │
│                                                                          │
│   ┌───────────┐   ┌───────────────┐   ┌─────────┐   ┌──────────────┐   │
│   │  Qdrant   │   │ Sentence      │   │  Redis  │   │  Ollama /    │   │
│   │  :6333    │   │ Transformers  │   │  :6379  │   │  OpenAI      │   │
│   │  vectors  │   │ embeddings    │   │  cache  │   │  LLM         │   │
│   └───────────┘   └───────────────┘   └─────────┘   └──────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Retrieval Pipeline — 7 Stages

| # | Stage | What It Does |
|---|-------|-------------|
| 1 | **Query Rewriting** | LLM generates 3-4 query variants to improve recall across different phrasings |
| 2 | **Dense Retrieval** | Encodes query with SentenceTransformers, finds top-20 semantically similar chunks via Qdrant |
| 3 | **BM25 Retrieval** | Keyword-based scoring on in-memory inverted index, finds top-20 exact-match chunks |
| 4 | **Score Fusion** | Reciprocal Rank Fusion (RRF) combines dense + sparse rankings in a scale-invariant way |
| 5 | **Cross-Encoder Reranking** | MS MARCO model scores query+document pairs for fine-grained relevance → top 5 |
| 6 | **Context Compression** | LLM extracts only query-relevant sentences from each chunk, reducing noise |
| 7 | **Prompt Injection** | Formats context with source labels and injects into the RAG system prompt |

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Minimum |
|-------------|---------|
| **Docker & Docker Compose** | v20+ / v2+ |
| **RAM** | 8 GB (for embedding + LLM models) |
| **Disk** | ~5 GB (Docker images + models) |

### Option 1: One-Command Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/smart_notes_assistant.git
cd smart_notes_assistant

# Make start script executable & run
chmod +x scripts/start.sh
./scripts/start.sh
```

This automatically:
1. ✅ Creates `.env` from the template
2. ✅ Starts all 5 Docker services (Qdrant, Redis, Ollama, API, Frontend)
3. ✅ Waits for health checks to pass
4. ✅ Pulls the Ollama LLM model (`llama3.2`)
5. ✅ Prints the service URLs

### Option 2: Docker Compose (Manual)

```bash
# 1. Copy environment config
cp .env.example .env

# 2. Start all services
docker compose up -d --build

# 3. Pull the LLM model (first time only, ~2GB download)
docker exec rag-ollama ollama pull llama3.2

# 4. Open in browser
open http://localhost:3000
```

### Option 3: Local Development (No Docker for backend)

```bash
# 1. Start infrastructure only
docker compose up -d qdrant redis ollama
docker exec rag-ollama ollama pull llama3.2

# 2. Start backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 3. Start frontend
cd ../frontend
npm install
npm run dev
```

---

## 📖 How to Use

### Step 1: Upload Your Documents

1. Click the **📎 paperclip icon** in the chat input bar, or click **"Upload documents"** in the sidebar
2. **Drag and drop** files into the upload zone, or click to browse
3. Supported formats: **PDF, TXT, Markdown, CSV, JSON** (up to 50MB each)
4. Optionally add **tags** (comma-separated) to organize your documents
5. Click upload — you'll see the processing status with chunk count

### Step 2: Ask Questions

1. Type your question in the chat input (or click a suggestion card)
2. The assistant will:
   - Search through all your uploaded documents
   - Find the most relevant passages
   - Generate an answer with **source citations**
3. Click **"X sources"** below any answer to see exactly which documents and passages were used

### Step 3: Manage Conversations

- **New Chat**: Click the **+** button in the top bar to start a fresh conversation
- **History**: Previous conversations appear in the sidebar — click to resume
- **Delete**: Hover over any conversation in the sidebar and click the trash icon

### Examples of What You Can Ask

| Query Type | Example |
|-----------|---------|
| **Summarization** | "Summarize the key points from my uploaded document" |
| **Specific Questions** | "What does the document say about machine learning algorithms?" |
| **Comparison** | "Compare the methodologies described in my uploaded papers" |
| **Extraction** | "List all the equations mentioned in the PDF" |
| **Analysis** | "What are the main arguments presented in chapter 3?" |

---

## ⚙️ Configuration

All settings are in `.env` (created from `.env.example`):

### LLM Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` (fully local) or `openai` (cloud API) |
| `OLLAMA_MODEL` | `llama3.2` | Any Ollama-supported model name |
| `OPENAI_API_KEY` | — | Required only if using `openai` provider |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |

### Models

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformers embedding model |

### Ports

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | `8000` | Backend API server |
| `FRONTEND_PORT` | `3000` | Next.js frontend |
| `QDRANT_PORT` | `6333` | Qdrant vector database |
| `REDIS_PORT` | `6379` | Redis cache |
| `OLLAMA_PORT` | `11434` | Ollama LLM server |

### Using OpenAI Instead of Ollama

```bash
# Edit .env:
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini

# Restart
docker compose up -d --build api
```

---

## 🔌 API Reference

Full interactive docs available at **http://localhost:8000/docs** (Swagger UI).

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns service status |
| `POST` | `/api/chat` | Send a query, receive streamed answer via SSE |
| `POST` | `/api/ingest` | Upload files for document ingestion |
| `POST` | `/api/ingest/directory` | Ingest all files from a server-side directory |

### Admin Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/admin/diagnostics` | System health, stats, and configuration |
| `GET` | `/api/admin/metrics` | Retrieval quality metrics (Recall@K, MRR) |
| `POST` | `/api/admin/metrics/reset` | Reset retrieval metrics |
| `GET` | `/api/admin/conversations` | List all conversation histories |
| `GET` | `/api/admin/documents` | List all indexed documents |
| `DELETE` | `/api/admin/documents/all` | Clear all documents from vector store |
| `DELETE` | `/api/admin/documents/{id}` | Delete a specific document by ID |

### Chat API Example

```bash
# Streaming chat request
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key points in my document?", "stream": true}'
```

### Upload API Example

```bash
# Upload a PDF
curl -X POST http://localhost:8000/api/ingest \
  -F "files=@my_document.pdf" \
  -F "tags=lecture,cs"
```

---

## 📁 Project Structure

```
smart_notes_assistant/
├── backend/                          # FastAPI backend
│   ├── main.py                       # App entry point + lifespan events
│   ├── Dockerfile                    # Multi-stage Docker build
│   ├── requirements.txt              # Python dependencies
│   └── app/
│       ├── config.py                 # Pydantic BaseSettings (env-driven)
│       ├── models.py                 # Request/response Pydantic schemas
│       ├── api/
│       │   ├── chat.py               # SSE streaming chat endpoint
│       │   ├── ingest.py             # File upload + ingestion pipeline
│       │   ├── admin.py              # Diagnostics, metrics, document mgmt
│       │   └── health.py             # Health check
│       ├── core/
│       │   └── cache.py              # Redis async caching layer
│       ├── embeddings/
│       │   └── service.py            # SentenceTransformers wrapper
│       ├── vectorstore/
│       │   └── qdrant_store.py       # Qdrant client + collection management
│       ├── ingestion/
│       │   ├── loader.py             # Multi-format document loaders
│       │   ├── chunker.py            # Recursive text splitter
│       │   └── pipeline.py           # Ingestion orchestrator
│       ├── retrieval/
│       │   ├── query_rewriter.py     # LLM-powered query expansion
│       │   ├── dense_retriever.py    # Vector similarity search
│       │   ├── sparse_retriever.py   # BM25 keyword search
│       │   ├── fusion.py             # Reciprocal Rank Fusion
│       │   ├── reranker.py           # Cross-encoder reranking
│       │   ├── compressor.py         # Context compression
│       │   └── pipeline.py           # 7-stage retrieval orchestrator
│       ├── llm/
│       │   ├── service.py            # Ollama + OpenAI dual-provider client
│       │   └── prompts.py            # Prompt templates
│       ├── evaluation/
│       │   └── metrics.py            # Recall@K, MRR tracking
│       └── utils/
│           ├── logging.py            # Structured JSON logging
│           └── exceptions.py         # Error hierarchy + handlers
│
├── frontend/                         # Next.js 14 frontend
│   ├── Dockerfile                    # Node.js Docker build
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── app/
│   │   ├── layout.tsx                # Root layout + metadata
│   │   ├── page.tsx                  # Main chat application
│   │   └── globals.css               # Dark theme design system
│   └── components/
│       ├── chat/
│       │   ├── MessageBubble.tsx      # Message display + markdown + citations
│       │   └── ChatInput.tsx          # Auto-resizing input with upload
│       ├── sidebar/
│       │   └── Sidebar.tsx            # Conversation history panel
│       └── upload/
│           └── DocumentUpload.tsx     # Drag-and-drop upload modal
│
├── docker-compose.yml                # 5-service orchestration
├── .env.example                      # Environment variable template
├── scripts/
│   └── start.sh                      # One-command startup script
└── data/
    ├── uploads/                      # Uploaded document storage
    └── knowledge/                    # Processed knowledge base
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Next.js 14, React 18, Tailwind CSS | ChatGPT-style dark UI with SSE streaming |
| **Backend** | Python 3.12, FastAPI, Pydantic v2 | Async REST API with type-safe validation |
| **Vector DB** | Qdrant (HNSW ANN) | Semantic similarity search on embeddings |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) | 384-dim dense embeddings |
| **Reranker** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) | Fine-grained relevance scoring |
| **Sparse Search** | rank_bm25 (BM25Okapi) | Keyword-based retrieval |
| **Cache** | Redis 7 | Query + embedding caching |
| **LLM** | Ollama (local) / OpenAI API | Response generation + query rewriting |
| **Container** | Docker, Docker Compose | One-command deployment |

---

## 🔧 Troubleshooting

<details>
<summary><b>Services won't start</b></summary>

```bash
# Check which services are running
docker compose ps

# View logs for a specific service
docker compose logs api --tail 50
docker compose logs frontend --tail 50

# Restart everything
docker compose down && docker compose up -d --build
```
</details>

<details>
<summary><b>Ollama model not found</b></summary>

```bash
# Pull the model manually
docker exec rag-ollama ollama pull llama3.2

# Verify it's downloaded
docker exec rag-ollama ollama list
```
</details>

<details>
<summary><b>Upload fails with "Field required"</b></summary>

Make sure you're using the correct API field name `files` (not `file`):
```bash
curl -X POST http://localhost:8000/api/ingest -F "files=@document.pdf"
```
</details>

<details>
<summary><b>Frontend shows blank page</b></summary>

```bash
# Clear browser cache: Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows)
# Or try an incognito window

# Check frontend logs
docker compose logs frontend --tail 20

# Rebuild frontend
docker compose up -d --build frontend
```
</details>

<details>
<summary><b>API returns 500 errors</b></summary>

```bash
# Check API logs for the actual error
docker compose logs api --tail 50

# Verify Qdrant and Redis are healthy
curl http://localhost:6333/healthz
docker exec rag-redis redis-cli ping
```
</details>

---

## 📈 Scaling Considerations

### Horizontal Scaling
- **API**: Stateless — scale with multiple containers behind a load balancer
- **Qdrant**: Supports native sharding and replication for 10M+ vectors
- **BM25**: Currently in-memory; migrate to Elasticsearch for multi-instance
- **Redis**: Use Redis Cluster for cache scaling

### Performance Tips
- **Embedding batching**: Documents are embedded in batches of 64 during ingestion
- **Reranker**: The cross-encoder (~50ms per query) is the slowest stage — consider GPU acceleration
- **Context compression**: Adds LLM latency per chunk; disable via `enable_compression` flag for speed
- **Model quantization**: Use ONNX Runtime or quantized models for 2-4x speedup

### Production Hardening
- Add authentication (API keys or OAuth)
- Rate limiting per user
- Request size limits
- Prometheus metrics export
- Centralized logging (ELK/Loki)
- TLS termination via reverse proxy (Nginx/Traefik)
- Automated Qdrant snapshot backups

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ using FastAPI, Next.js, Qdrant, and Ollama
</p>
