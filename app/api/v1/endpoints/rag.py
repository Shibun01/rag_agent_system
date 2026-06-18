from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from app.models.schemas import RAGQueryRequest, RAGQueryResponse, RAGStrategy
from app.core.rag.naive_rag import naive_rag
from app.core.rag.advanced_rag import advanced_rag
from app.core.rag.corrective_rag import corrective_rag
from app.core.rag.self_rag import self_rag
from app.core.rag.hyde_rag import hyde_rag
from app.core.rag.multiquery_rag import multi_query_rag
from app.core.rag.graph_rag import graph_rag

router = APIRouter()

_STRATEGY_MAP = {
    RAGStrategy.naive:       naive_rag,
    RAGStrategy.advanced:    advanced_rag,
    RAGStrategy.corrective:  corrective_rag,
    RAGStrategy.self_rag:    self_rag,
    RAGStrategy.hyde:        hyde_rag,
    RAGStrategy.multi_query: multi_query_rag,
    RAGStrategy.graph_rag:   graph_rag,
}


@router.post("/query", response_model=RAGQueryResponse, summary="Run a RAG query")
async def rag_query(req: RAGQueryRequest):
    """
    Execute a query using one of the available RAG strategies:

    | Strategy      | Description |
    |---------------|-------------|
    | `naive`       | Embed → retrieve → generate |
    | `advanced`    | Hybrid search + reranking |
    | `corrective`  | Grade docs → rewrite → supplement (CRAG) |
    | `self_rag`    | LLM decides when & whether to retrieve |
    | `hyde`        | Embed hypothetical answer doc for retrieval |
    | `multi_query` | Multiple query rephrasings merged |
    | `graph_rag`   | Knowledge graph + vector retrieval |
    """
    handler = _STRATEGY_MAP.get(req.strategy)
    if not handler:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {req.strategy}")
    try:
        result = await handler(
            query=req.query,
            collection=req.collection_name,
            top_k=req.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RAGQueryResponse(
        answer=result.get("answer", ""),
        strategy=req.strategy,
        sources=result.get("sources", []),
        metadata={k: v for k, v in result.items() if k not in ("answer", "sources")},
    )


@router.get("/strategies", summary="List available RAG strategies")
async def list_strategies():
    return {
        "strategies": [
            {"name": s.value, "description": desc}
            for s, desc in {
                RAGStrategy.naive:       "Baseline: embed → retrieve → generate",
                RAGStrategy.advanced:    "Hybrid search + cross-encoder reranking",
                RAGStrategy.corrective:  "CRAG: grade relevance, rewrite if needed",
                RAGStrategy.self_rag:    "Self-RAG: adaptive retrieval with LLM critique tokens",
                RAGStrategy.hyde:        "HyDE: hypothetical document embedding",
                RAGStrategy.multi_query: "Generate multiple query variants, merge results",
                RAGStrategy.graph_rag:   "Knowledge graph traversal + vector search",
            }.items()
        ]
    }
