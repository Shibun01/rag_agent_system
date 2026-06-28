from __future__ import annotations

import inspect
import time

from fastapi import APIRouter, Depends, HTTPException
from app.api.dependencies import get_tenant_id
from app.core.tenancy import scope_collection_name
from app.models.schemas import (
    RAGCompareRequest,
    RAGCompareResponse,
    RAGCompareResult,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGStrategy,
)
from app.core.rag.naive_rag import naive_rag
from app.core.rag.advanced_rag import advanced_rag
from app.core.rag.corrective_rag import corrective_rag
from app.core.rag.self_rag import self_rag
from app.core.rag.hyde_rag import hyde_rag
from app.core.rag.multiquery_rag import multi_query_rag
from app.core.rag.graph_rag import graph_rag
from app.services.analytics import log_query

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
async def _run_strategy(
    req: RAGQueryRequest | RAGCompareRequest,
    strategy: RAGStrategy,
    tenant_id: str,
) -> dict:
    handler = _STRATEGY_MAP.get(strategy)
    if not handler:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")

    kwargs = {
        "query": req.query,
        "collection": scope_collection_name(req.collection_name, tenant_id),
        "top_k": req.top_k,
    }
    signature = inspect.signature(handler)
    if "filters" in signature.parameters:
        kwargs["filters"] = req.filters
    if "rerank" in signature.parameters:
        kwargs["rerank"] = req.rerank
    return await handler(**kwargs)


@router.post("/query", response_model=RAGQueryResponse, summary="Run a RAG query")
async def rag_query(req: RAGQueryRequest, tenant_id: str = Depends(get_tenant_id)):
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
    started = time.perf_counter()
    try:
        result = await _run_strategy(req, req.strategy, tenant_id)
        latency_ms = (time.perf_counter() - started) * 1000
        sources = result.get("sources", []) if req.include_sources else []
        await log_query(
            tenant_id=tenant_id,
            query=req.query,
            strategy=req.strategy.value,
            collection_name=req.collection_name,
            latency_ms=latency_ms,
            status="success",
            source_count=len(sources),
            answer=result.get("answer", ""),
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - started) * 1000
        await log_query(
            tenant_id=tenant_id,
            query=req.query,
            strategy=req.strategy.value,
            collection_name=req.collection_name,
            latency_ms=latency_ms,
            status="error",
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))

    return RAGQueryResponse(
        answer=result.get("answer", ""),
        strategy=req.strategy,
        sources=sources,
        metadata={k: v for k, v in result.items() if k not in ("answer", "sources")},
    )


@router.post("/compare", response_model=RAGCompareResponse, summary="Compare RAG strategies")
async def compare_strategies(req: RAGCompareRequest, tenant_id: str = Depends(get_tenant_id)):
    results = []
    for strategy in req.strategies:
        started = time.perf_counter()
        try:
            result = await _run_strategy(req, strategy, tenant_id)
            latency_ms = (time.perf_counter() - started) * 1000
            sources = result.get("sources", [])
            metadata = {k: v for k, v in result.items() if k not in ("answer", "sources")}
            await log_query(
                tenant_id=tenant_id,
                query=req.query,
                strategy=strategy.value,
                collection_name=req.collection_name,
                latency_ms=latency_ms,
                status="success",
                source_count=len(sources),
                answer=result.get("answer", ""),
            )
            results.append(RAGCompareResult(
                strategy=strategy,
                answer=result.get("answer", ""),
                sources=sources,
                metadata=metadata,
                latency_ms=latency_ms,
                source_count=len(sources),
            ))
        except Exception as e:
            latency_ms = (time.perf_counter() - started) * 1000
            await log_query(
                tenant_id=tenant_id,
                query=req.query,
                strategy=strategy.value,
                collection_name=req.collection_name,
                latency_ms=latency_ms,
                status="error",
                error=str(e),
            )
            results.append(RAGCompareResult(
                strategy=strategy,
                latency_ms=latency_ms,
                status="error",
                error=str(e),
            ))
    return RAGCompareResponse(
        query=req.query,
        collection_name=req.collection_name,
        results=results,
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
