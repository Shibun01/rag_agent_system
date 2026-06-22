# 🚀 RAG Agent System

A production-grade **Retrieval-Augmented Generation (RAG)** and **AI Agent orchestration platform** built with **FastAPI**, **Azure OpenAI**, and **ChromaDB**.

> Enterprise-ready RAG and Agent orchestration system showcasing advanced retrieval, reasoning, and multi-agent execution patterns.

This system enables:
- Advanced document retrieval
- Context-aware AI responses
- Multi-agent workflows
- Production-ready streaming APIs

Designed for enterprise-grade use cases such as:
- Knowledge assistants  
- Document intelligence  
- Enterprise copilots  
- Multi-agent automation systems  

---

## ✨ Features

### Advanced RAG Pipelines
Supports multiple retrieval strategies for different workloads:

- Naive RAG
- Advanced RAG
- Corrective RAG (CRAG)
- Self-RAG
- HyDE
- Multi-Query RAG
- GraphRAG

---

### Agentic AI Workflows
Supports modern agent orchestration patterns:

- ReAct Agent
- Plan-and-Execute Agent
- Reflection Agent
- Supervisor Agent
- RAG Agent
- Multi-Agent Collaboration

---

### Intelligent Document Processing
- PDF / DOCX / TXT ingestion
- Semantic chunking
- Embedding generation
- Vector indexing with ChromaDB

---

### Production Features
- Async APIs with FastAPI
- SSE streaming support
- Azure OpenAI integration
- Memory management
- Tool registry support
- Dockerized deployment
- Structured logging

---

# 🏗️ Architecture

```bash
rag-agent-system/
├── app/
│   ├── api/                # REST endpoints
│   ├── core/               # RAG + Agents logic
│   ├── services/           # External service integrations
│   ├── config/             # Settings & logging
│   └── models/             # Request/response schemas
│
├── docker/                 # Docker setup
├── tests/                  # Test cases
├── requirements.txt
└── .env.example
```

---

## System Flow

```text
Documents → Chunking → Embeddings → Vector Store
                                   ↓
User Query → RAG / Agents → Retrieval → LLM → Response
```

---

# 🧠 Supported RAG Strategies

| Strategy | Description |
|----------|-------------|
| Naive RAG | Basic retrieve + generate |
| Advanced RAG | Hybrid retrieval + reranking |
| Corrective RAG | Query correction + relevance grading |
| Self-RAG | Self-evaluation based retrieval |
| HyDE | Hypothetical document embeddings |
| Multi-Query RAG | Query expansion for better recall |
| GraphRAG | Graph-based multi-hop reasoning |

---

# 🤖 Supported Agent Patterns

| Agent Type | Description |
|------------|-------------|
| ReAct | Reasoning + tool usage loop |
| Plan-and-Execute | Planning before execution |
| Reflection | Self-critique and refinement |
| Supervisor | Task routing to sub-agents |
| RAG Agent | Retrieval-first agent |
| Multi-Agent | Parallel specialized agents |

---

# 📦 Chunking Strategies

| Strategy | Description |
|----------|-------------|
| Recursive | Recursive split with overlap |
| Semantic | Embedding similarity-based splitting |
| Parent-Child | Parent-child chunk hierarchy |
| Sentence Window | Sentence retrieval with context window |

---

# ⚙️ Tech Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI |
| LLM | Azure OpenAI |
| Vector DB | ChromaDB |
| Agent Framework | LangChain / LangGraph |
| Reranking | FlashRank / Cohere |
| Embeddings | Sentence Transformers |
| Document Parsing | PyMuPDF |

---

# 🚀 Quick Start

## 1. Clone Repository

```bash
git clone <repo-url>
cd rag-agent-system
```

---

## 2. Configure Environment

```bash
cp .env.example .env
```

Configure:

```env
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT=
CHROMA_DB_PATH=
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run Locally

```bash
uvicorn app.main:app --reload --port 8000
```

Open Swagger UI:

```bash
http://localhost:8000/docs
```

---

## 5. Run with Docker

```bash
cd docker
docker-compose up --build
```

---

# 📡 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/documents/ingest` | Upload and process documents |
| `/rag/query` | Query using selected RAG strategy |
| `/agents/run` | Execute agent workflows |
| `/chat` | Standard chat |
| `/chat/stream` | Streaming chat |

---

# Example Usage

## Document Upload

```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
-F "file=@report.pdf" \
-F "collection_name=mydata" \
-F "chunk_strategy=sentence_window"
```

---

## RAG Query

```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
-H "Content-Type: application/json" \
-d '{
  "query": "What are the key findings?",
  "strategy": "corrective",
  "collection_name": "mydata",
  "top_k": 5
}'
```

---

## Agent Execution

```bash
curl -X POST http://localhost:8000/api/v1/agents/run \
-H "Content-Type: application/json" \
-d '{
  "task": "Summarize document insights",
  "agent_type": "rag_agent",
  "max_iterations": 8
}'
```

---

# 🔥 Use Cases

- Enterprise AI copilots  
- AI knowledge assistants  
- Customer support bots  
- Research assistants  
- Multi-agent automation systems  
- Internal documentation search  

---

# Future Enhancements

- Multi-modal RAG  
- Graph database integration  
- Agent memory persistence  
- Evaluation dashboards  
- Observability & tracing  

---

# 📄 License

MIT License
