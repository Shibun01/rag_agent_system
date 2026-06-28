# RAG Agent System

Production-grade **Retrieval-Augmented Generation** and **Agents** API built with
**FastAPI** + **Azure OpenAI** + **ChromaDB**.

---

## Architecture Overview

```
rag-agent-system/
├── app/
│   ├── main.py                        # FastAPI app, lifespan, CORS
│   ├── config/                        # Settings (pydantic-settings) + structured logging
│   ├── api/v1/endpoints/
│   │   ├── rag.py                     # POST /api/v1/rag/query
│   │   ├── agents.py                  # POST /api/v1/agents/run
│   │   ├── documents.py               # POST /api/v1/documents/ingest
│   │   └── chat.py                    # POST /api/v1/chat  (+ /stream SSE)
│   ├── core/
│   │   ├── rag/                       # All RAG strategies (see below)
│   │   ├── agents/                    # All agent patterns (see below)
│   │   ├── chunking/strategies.py     # 4 chunking strategies
│   │   ├── memory/conversation.py     # Short-term + long-term memory
│   │   └── tools/registry.py         # Tool registry (calculator, web search, …)
│   ├── services/
│   │   ├── azure_openai.py            # Async Azure OpenAI client wrapper
│   │   ├── vector_store.py            # ChromaDB vector store abstraction
│   │   ├── reranker.py                # FlashRank (local) / Cohere reranker
│   │   └── document_processor.py     # PDF / DOCX / TXT extraction
│   └── models/schemas.py             # Pydantic request/response models
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
├── .env.example
└── requirements.txt
```

---

## RAG Strategies

| Strategy | File | Key Concept |
|---|---|---|
| **Naive RAG** | `core/rag/naive_rag.py` | Embed → top-k retrieve → generate |
| **Advanced RAG** | `core/rag/advanced_rag.py` | Hybrid search + RRF fusion + cross-encoder reranking |
| **Corrective RAG (CRAG)** | `core/rag/corrective_rag.py` | Grade relevance → rewrite query if irrelevant → supplement |
| **Self-RAG** | `core/rag/self_rag.py` | LLM emits RETRIEVE / ISREL / ISSUP critique tokens |
| **HyDE** | `core/rag/hyde_rag.py` | Generate hypothetical answer doc, embed that for retrieval |
| **Multi-Query RAG** | `core/rag/multiquery_rag.py` | Generate N query rephrasings → merge deduplicated results |
| **GraphRAG** | `core/rag/graph_rag.py` | Extract entity graph → traverse for multi-hop reasoning |

---

## Agent Patterns

| Agent | File | Key Concept |
|---|---|---|
| **ReAct** | `core/agents/react_agent.py` | Thought → Action → Observation loop |
| **Plan-and-Execute** | `core/agents/plan_execute_agent.py` | Step-by-step plan → execute each + optional replan |
| **Reflection** | `core/agents/reflection_agent.py` | Generate → critique → refine (N rounds) |
| **Supervisor** | `core/agents/supervisor_agent.py` | LLM routes task to the best sub-agent |
| **RAG Agent** | `core/agents/rag_agent.py` | ReAct where primary tool is semantic document retrieval |
| **Multi-Agent** | `core/agents/multi_agent.py` | Decompose → parallel specialists → synthesize |

---

## Chunking Strategies

| Strategy | Description |
|---|---|
| `recursive` | Paragraph → sentence → word recursive splitting with overlap |
| `semantic` | Cosine-distance breakpoints between sentence embeddings |
| `parent_child` | Large parent chunks index; small child chunks retrieved |
| `sentence_window` | Single-sentence retrieval unit with ±N sentence window for context |

---

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Fill in AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, etc.
```

Key production-oriented settings now available in `.env`:

```env
AUTH_ENABLED=false
API_KEYS=
DEFAULT_TENANT_ID=default
REQUIRE_TENANT_HEADER=false
CORS_ALLOW_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
CONVERSATION_DB_PATH=./data/conversations.sqlite3
```

- `AUTH_ENABLED` + `API_KEYS` enable API-key protection with the `X-API-Key` header.
- `DEFAULT_TENANT_ID` and `REQUIRE_TENANT_HEADER` control tenant scoping via `X-Tenant-ID`.
- `CONVERSATION_DB_PATH` persists chat memory across restarts.
- `CORS_ALLOW_*` settings tighten browser access for the frontend.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000/docs> for the interactive Swagger UI.

### 4. Run the Streamlit Studio

### 4. Run the Vite Dashboard

```bash
cd frontend
npm install
npm run dev
```

The Tailwind dashboard provides tenant-aware document ingestion, RAG querying,
strategy comparison, grounded chat, agent runs, and analytics.

### 5. Run the Streamlit Studio

```bash
streamlit run frontend_stremlit/streamlit_app.py
```

The studio provides document ingestion, collection management, RAG querying,
strategy comparison, query logs, and feedback capture in one UI.

### 6. Docker

```bash
cd docker
docker-compose up --build
```

---

## API Quick Reference

### Upload a document

```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -H "X-Tenant-ID: demo" \
  -F "file=@report.pdf" \
  -F "collection_name=mydata" \
  -F "chunk_strategy=sentence_window"
```

### RAG query

```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "X-Tenant-ID: demo" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key findings?",
    "strategy": "corrective",
    "collection_name": "mydata",
    "top_k": 5
  }'
```

### Run an agent

```bash
curl -X POST http://localhost:8000/api/v1/agents/run \
  -H "X-Tenant-ID: demo" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Summarize the main themes in the uploaded documents and calculate total word count.",
    "agent_type": "rag_agent",
    "max_iterations": 8,
    "tools": ["search", "calculator"],
    "use_memory": true
  }'
```

### Streaming chat

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "X-Tenant-ID: demo" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is RAG?"}],
    "rag_strategy": "advanced",
    "collection_name": "mydata"
  }'
```

If API-key auth is enabled, add `-H "X-API-Key: <your-key>"` to each request.

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | Async API framework |
| `openai` | Azure OpenAI SDK |
| `chromadb` | Local persistent vector store |
| `langchain` / `langgraph` | Orchestration utilities |
| `flashrank` | Local cross-encoder reranking |
| `sentence-transformers` | Semantic chunking embeddings |
| `pymupdf` | PDF text extraction |
