import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest, ChatResponse, ChatMessage, RAGStrategy
from app.services.azure_openai import get_azure_openai_client
from app.config.settings import get_settings
from app.core.memory.conversation import add_message, build_context_messages, summarize_and_compress
from app.core.rag.advanced_rag import advanced_rag

router = APIRouter()
settings = get_settings()


@router.post("/", response_model=ChatResponse, summary="Multi-turn chat with optional RAG")
async def chat(req: ChatRequest):
    """
    Stateful multi-turn chat. Pass a `session_id` to maintain conversation history.
    Set `rag_strategy` to ground answers in your documents.
    """
    session_id = req.session_id or str(uuid.uuid4())

    # Store incoming user message
    last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
    if last_user:
        add_message(session_id, "user", last_user.content)

    # Optionally compress long history
    await summarize_and_compress(session_id)

    # Build message history with optional long-term summary
    history = build_context_messages(session_id)

    # RAG augmentation
    rag_context = ""
    sources = []
    if req.rag_strategy and last_user:
        rag_result = await advanced_rag(
            query=last_user.content,
            collection=req.collection_name,
        )
        rag_context = "\n\n".join(s["content"] for s in rag_result.get("sources", []))
        sources = rag_result.get("sources", [])

    if rag_context:
        history.insert(0, {
            "role": "system",
            "content": f"Use the following retrieved context to assist the user:\n\n{rag_context}",
        })

    client = get_azure_openai_client()
    response = await client.chat.completions.create(
        model=settings.azure_openai_deployment_name,
        messages=history,
        temperature=0.7,
    )
    answer = response.choices[0].message.content

    add_message(session_id, "assistant", answer)

    return ChatResponse(
        message=ChatMessage(role="assistant", content=answer),
        session_id=session_id,
        sources=sources,
    )


@router.post("/stream", summary="Streaming chat")
async def chat_stream(req: ChatRequest):
    """Server-Sent Events streaming chat."""
    session_id = req.session_id or str(uuid.uuid4())
    last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
    if last_user:
        add_message(session_id, "user", last_user.content)

    history = build_context_messages(session_id)
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
        add_message(session_id, "assistant", full_text)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
