import inspect
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.api.dependencies import get_tenant_id
from app.core.tenancy import scope_collection_name
from app.models.schemas import ChatRequest, ChatResponse, ChatMessage, RAGStrategy
from app.services.azure_openai import get_azure_openai_client
from app.config.settings import get_settings
from app.core.memory.conversation import add_message, build_context_messages, summarize_and_compress
from app.core.rag.naive_rag import naive_rag
from app.core.rag.advanced_rag import advanced_rag
from app.core.rag.corrective_rag import corrective_rag
from app.core.rag.self_rag import self_rag
from app.core.rag.hyde_rag import hyde_rag
from app.core.rag.multiquery_rag import multi_query_rag
from app.core.rag.graph_rag import graph_rag

router = APIRouter()
settings = get_settings()

_STRATEGY_MAP = {
    RAGStrategy.naive: naive_rag,
    RAGStrategy.advanced: advanced_rag,
    RAGStrategy.corrective: corrective_rag,
    RAGStrategy.self_rag: self_rag,
    RAGStrategy.hyde: hyde_rag,
    RAGStrategy.multi_query: multi_query_rag,
    RAGStrategy.graph_rag: graph_rag,
}


async def _prepare_history(req: ChatRequest, session_id: str, tenant_id: str) -> tuple[list[dict], list[dict]]:
    last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
    if last_user:
        add_message(session_id, "user", last_user.content, tenant_id=tenant_id)

    await summarize_and_compress(session_id, tenant_id=tenant_id)
    history = build_context_messages(session_id, tenant_id=tenant_id)
    sources: list[dict] = []

    if req.rag_strategy and last_user:
        handler = _STRATEGY_MAP.get(req.rag_strategy)
        if not handler:
            raise HTTPException(status_code=400, detail=f"Unknown RAG strategy: {req.rag_strategy}")

        kwargs = {
            "query": last_user.content,
            "collection": scope_collection_name(req.collection_name, tenant_id),
            "top_k": settings.top_k,
        }
        signature = inspect.signature(handler)
        if "rerank" in signature.parameters:
            kwargs["rerank"] = True

        rag_result = await handler(**kwargs)
        sources = rag_result.get("sources", [])
        rag_context = "\n\n".join(source.get("content", "") for source in sources if source.get("content"))
        if rag_context:
            history.insert(0, {
                "role": "system",
                "content": f"Use the following retrieved context to assist the user:\n\n{rag_context}",
            })

    return history, sources


@router.post("/", response_model=ChatResponse, summary="Multi-turn chat with optional RAG")
async def chat(req: ChatRequest, tenant_id: str = Depends(get_tenant_id)):
    """
    Stateful multi-turn chat. Pass a `session_id` to maintain conversation history.
    Set `rag_strategy` to ground answers in your documents.
    """
    session_id = req.session_id or str(uuid.uuid4())
    history, sources = await _prepare_history(req, session_id, tenant_id)

    client = get_azure_openai_client()
    response = await client.chat.completions.create(
        model=settings.azure_openai_deployment_name,
        messages=history,
        temperature=0.7,
    )
    answer = response.choices[0].message.content

    add_message(session_id, "assistant", answer, tenant_id=tenant_id)

    return ChatResponse(
        message=ChatMessage(role="assistant", content=answer),
        session_id=session_id,
        sources=sources,
    )


@router.post("/stream", summary="Streaming chat")
async def chat_stream(req: ChatRequest, tenant_id: str = Depends(get_tenant_id)):
    """Server-Sent Events streaming chat."""
    session_id = req.session_id or str(uuid.uuid4())
    history, _ = await _prepare_history(req, session_id, tenant_id)
    client = get_azure_openai_client()

    async def event_generator():
        stream = await client.chat.completions.create(
            model=settings.azure_openai_deployment_name,
            messages=history,
            stream=True,
        )
        full_text = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_text += delta
            yield f"data: {delta}\n\n"
        add_message(session_id, "assistant", full_text, tenant_id=tenant_id)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"X-Session-ID": session_id, "Cache-Control": "no-cache"},
    )
