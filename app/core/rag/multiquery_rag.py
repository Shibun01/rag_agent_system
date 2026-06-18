"""
Multi-Query RAG — generates multiple rephrasings of the user query, retrieves
documents for each, and merges results via deduplication.

Addresses the problem that a single query may not capture all semantically
relevant documents due to embedding space coverage gaps.
"""
from __future__ import annotations
import asyncio
import json
from app.services.azure_openai import chat_completion
from app.services.vector_store import get_vector_store, SearchResult

MULTI_QUERY_PROMPT = """You are an AI that generates multiple search queries to retrieve
relevant information. Given a question, generate {n} different rephrasings of the question
that approach it from different angles. Output a JSON array of strings only.

Question: {question}

Output (JSON array):"""


async def _generate_queries(question: str, n: int = 3) -> list[str]:
    msg = await chat_completion(
        [{"role": "user", "content": MULTI_QUERY_PROMPT.format(question=question, n=n)}],
        temperature=0.7,
    )
    try:
        queries = json.loads(msg.content)
        return [question] + [q for q in queries if isinstance(q, str)]
    except Exception:
        return [question]


async def multi_query_rag(
    query: str,
    collection: str = "default",
    top_k: int = 5,
    num_queries: int = 3,
) -> dict:
    store = get_vector_store()

    # Step 1: Generate query variants
    queries = await _generate_queries(query, num_queries)

    # Step 2: Retrieve for each query concurrently
    tasks = [store.similarity_search(q, collection, top_k) for q in queries]
    all_results: list[list[SearchResult]] = await asyncio.gather(*tasks)

    # Step 3: Merge & deduplicate by document ID
    seen: set[str] = set()
    merged: list[SearchResult] = []
    for results in all_results:
        for r in results:
            if r.id not in seen:
                seen.add(r.id)
                merged.append(r)

    # Step 4: Generate answer
    context = "\n\n".join(f"[Source {i+1}]: {r.content}" for i, r in enumerate(merged))
    msg = await chat_completion([
        {"role": "system", "content": "Answer the question using the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ])

    return {
        "answer": msg.content,
        "queries_used": queries,
        "unique_sources_retrieved": len(merged),
        "sources": [{"id": r.id, "content": r.content, "score": r.score} for r in merged],
    }
