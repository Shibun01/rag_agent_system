from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


class RAGStrategy(str, Enum):
    naive = "naive"
    advanced = "advanced"
    corrective = "corrective"  # CRAG
    self_rag = "self_rag"
    graph_rag = "graph_rag"
    hyde = "hyde"
    multi_query = "multi_query"


class AgentType(str, Enum):
    react = "react"
    plan_execute = "plan_execute"
    reflection = "reflection"
    supervisor = "supervisor"
    rag_agent = "rag_agent"
    multi_agent = "multi_agent"


# ── RAG ──────────────────────────────────────────────────────────────────────
class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    strategy: RAGStrategy = RAGStrategy.advanced
    top_k: int = Field(5, ge=1, le=20)
    collection_name: str = "default"
    filters: Optional[dict] = None
    rerank: bool = True
    include_sources: bool = True


class RAGQueryResponse(BaseModel):
    answer: str
    strategy: RAGStrategy
    sources: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class RAGCompareRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    strategies: list[RAGStrategy] = Field(
        default_factory=lambda: [RAGStrategy.naive, RAGStrategy.advanced, RAGStrategy.multi_query],
        min_length=1,
        max_length=7,
    )
    top_k: int = Field(5, ge=1, le=20)
    collection_name: str = "default"
    filters: Optional[dict] = None
    rerank: bool = True


class RAGCompareResult(BaseModel):
    strategy: RAGStrategy
    answer: str = ""
    sources: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    latency_ms: float
    source_count: int = 0
    status: Literal["success", "error"] = "success"
    error: Optional[str] = None


class RAGCompareResponse(BaseModel):
    query: str
    collection_name: str
    results: list[RAGCompareResult]


# ── Agents ────────────────────────────────────────────────────────────────────
class AgentRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=4000)
    agent_type: AgentType = AgentType.react
    max_iterations: int = Field(10, ge=1, le=30)
    tools: list[str] = Field(default_factory=lambda: ["search", "calculator"])
    session_id: Optional[str] = None
    use_memory: bool = True


class AgentResponse(BaseModel):
    result: str
    agent_type: AgentType
    steps: list[dict] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)
    iterations: int = 0


# ── Documents ─────────────────────────────────────────────────────────────────
class DocumentIngestRequest(BaseModel):
    collection_name: str = "default"
    chunk_strategy: Literal["recursive", "semantic", "parent_child", "sentence_window"] = "recursive"
    chunk_size: int = Field(512, ge=128, le=4096)
    chunk_overlap: int = Field(64, ge=0, le=512)
    metadata: dict = Field(default_factory=dict)


class DocumentIngestResponse(BaseModel):
    document_id: str
    chunks_created: int
    collection_name: str
    status: str = "success"


class CollectionSummary(BaseModel):
    name: str
    chunk_count: int = 0
    document_count: int = 0
    sources: list[str] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    document_id: str
    source: str
    chunk_count: int
    metadata: dict = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    id: str
    content: str
    metadata: dict = Field(default_factory=dict)


# ── Chat ──────────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    session_id: Optional[str] = None
    rag_strategy: Optional[RAGStrategy] = None
    collection_name: str = "default"
    stream: bool = False


class ChatResponse(BaseModel):
    message: ChatMessage
    session_id: str
    sources: list[dict] = Field(default_factory=list)


# ── Analytics ────────────────────────────────────────────────────────────────
class QueryLogResponse(BaseModel):
    id: int
    created_at: str
    tenant_id: str = "default"
    query: str
    strategy: str
    collection_name: str
    latency_ms: float
    status: str
    source_count: int = 0
    answer_preview: str = ""
    error: Optional[str] = None


class FeedbackRequest(BaseModel):
    query_log_id: Optional[int] = None
    rating: Literal["up", "down"]
    comment: str = Field("", max_length=2000)
    query: Optional[str] = None
    strategy: Optional[str] = None
    collection_name: str = "default"


class FeedbackResponse(BaseModel):
    id: int
    created_at: str
    tenant_id: str = "default"
    query_log_id: Optional[int] = None
    rating: str
    comment: str = ""
    query: Optional[str] = None
    strategy: Optional[str] = None
    collection_name: str = "default"
