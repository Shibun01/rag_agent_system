# 🧠 RAG Agent System

> Production-grade Retrieval-Augmented Generation (RAG) and Agentic AI platform built with **FastAPI**, **Azure OpenAI**, **ChromaDB**, and **LangGraph**.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Async-green?logo=fastapi" />
  <img src="https://img.shields.io/badge/Azure-OpenAI-blue" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-purple" />
  <img src="https://img.shields.io/badge/LangGraph-Agentic-orange" />
  <img src="https://img.shields.io/badge/License-MIT-success" />
</p>

---

## 🚀 Overview

An enterprise-grade **RAG + Agent Orchestration API** designed to support advanced retrieval, multi-step reasoning, tool execution, and multi-agent collaboration.

This system provides:

- 🔍 Advanced retrieval pipelines  
- 🤖 Agentic task execution  
- 🧠 Memory-aware conversations  
- 📡 Streaming chat APIs  
- 📄 Intelligent document ingestion  

Built for:
- Enterprise copilots
- Knowledge assistants
- Research systems
- AI-powered internal search
- Agentic workflows

---

# 🏗️ System Architecture

```bash
rag-agent-system/
├── app/
│   ├── api/v1/endpoints/
│   ├── core/
│   │   ├── rag/
│   │   ├── agents/
│   │   ├── chunking/
│   │   ├── memory/
│   │   └── tools/
│   ├── services/
│   ├── config/
│   └── models/
│
├── docker/
├── tests/
└── requirements.txt
```

---

## ⚙️ Request Lifecycle

```text
                    ┌──────────────────────┐
                    │   User Query / Task  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Query Understanding  │
                    └──────────┬───────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
     ┌──────────────────┐              ┌──────────────────┐
     │   RAG Pipeline   │              │   Agent Pipeline │
     └──────────────────┘              └──────────────────┘
              │                                 │
              └──────────────┬──────────────────┘
                             ▼
                    ┌──────────────────────┐
                    │ Azure OpenAI / LLM   │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Final Response       │
                    └──────────────────────┘
```

---

# 🔍 RAG Strategies

Supports multiple retrieval strategies optimized for different workloads.

| Strategy | Description |
|----------|-------------|
| Naive RAG | Embed → Retrieve → Generate |
| Advanced RAG | Hybrid retrieval + RRF + reranking |
| CRAG | Corrective retrieval with query rewrite |
| Self-RAG | LLM-guided retrieval evaluation |
| HyDE | Hypothetical document embeddings |
| Multi-Query RAG | Query expansion |
| GraphRAG | Multi-hop graph traversal |

---

## RAG Flow

```text
Documents
   │
   ▼
Chunking
   │
   ▼
Embedding Generation
   │
   ▼
ChromaDB Storage
   │
   ▼
Retriever → Reranker → LLM → Response
```

---

# 🤖 Agent Patterns

Supports advanced agentic execution patterns.

| Agent | Description |
|-------|-------------|
| ReAct | Thought → Action → Observation |
| Plan-and-Execute | Plan first, execute stepwise |
| Reflection | Self-correction loops |
| Supervisor | Dynamic task routing |
| RAG Agent | Retrieval-first reasoning |
| Multi-Agent | Parallel specialist collaboration |

---

## Multi-Agent Flow

```text
Task
 │
 ▼
Supervisor Agent
 │
 ├── Research Agent
 ├── Retrieval Agent
 ├── Calculator Agent
 └── Summarizer Agent
          │
          ▼
   Final Synthesized Response
```

---

# 📦 Chunking Strategies

| Strategy | Description |
|----------|-------------|
| Recursive | Recursive text splitting |
| Semantic | Embedding similarity based |
| Parent Child | Parent-child retrieval |
| Sentence Window | Context-aware sentence retrieval |

---

# 🛠️ Core Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI |
| LLM | Azure OpenAI |
| Vector DB | ChromaDB |
| Agent Framework | LangChain / LangGraph |
| Reranker | FlashRank / Cohere |
| Parsing | PyMuPDF |

---

# 📡 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/documents/ingest` | Upload documents |
| `/rag/query` | Execute RAG queries |
| `/agents/run` | Run agents |
| `/chat` | Chat API |
| `/chat/stream` | SSE streaming |

---

# 🚀 Quick Start

## Setup

```bash
cp .env.example .env
```

Fill:

```env
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT=
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
uvicorn app.main:app --reload --port 8000
```

Swagger UI:

```bash
http://localhost:8000/docs
```

---

# Example APIs

### Document Ingestion

```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
-F "file=@report.pdf" \
-F "collection_name=mydata"
```

### RAG Query

```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
-H "Content-Type: application/json" \
-d '{
 "query":"What are the key findings?",
 "strategy":"advanced"
}'
```

### Agent Run

```bash
curl -X POST http://localhost:8000/api/v1/agents/run \
-H "Content-Type: application/json" \
-d '{
 "task":"Analyze and summarize uploaded docs",
 "agent_type":"rag_agent"
}'
```

---

# 🔥 Key Features

- 7 RAG strategies  
- 6 agent architectures  
- Streaming responses  
- Tool integration  
- Memory support  
- Production-ready deployment  

---

# 🎯 Use Cases

- Enterprise AI Copilot  
- Document Intelligence  
- Internal Knowledge Search  
- AI Research Assistant  
- Agentic Automation  

---

# 📄 License

MIT License
