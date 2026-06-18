"""
Naive RAG — the baseline pattern.
Pipeline: embed query → retrieve top-k → stuff into prompt → generate answer.
"""
from app.services.azure_openai import chat_completion
from app.services.vector_store import get_vector_store, SearchResult

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using ONLY the
provided context. If the answer is not in the context, say "I don't know"."""


async def naive_rag(
    query: str,
    collection: str = "default",
    top_k: int = 5,
) -> dict:
    # 1. Retrieve
    store = get_vector_store()
    results: list[SearchResult] = await store.similarity_search(query, collection, top_k)

    # 2. Build context
    context = "\n\n".join(
        f"[Source {i+1}]: {r.content}" for i, r in enumerate(results)
    )

    # 3. Generate
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    message = await chat_completion(messages)

    return {
        "answer": message.content,
        "sources": [{"id": r.id, "content": r.content, "score": r.score} for r in results],
    }
