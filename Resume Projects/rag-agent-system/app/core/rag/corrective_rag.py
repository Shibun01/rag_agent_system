"""
Corrective RAG (CRAG) — from the paper "Corrective Retrieval Augmented Generation" (2024).

Key idea:
  1. Retrieve documents.
  2. Grade each document for relevance to the query.
  3. If all documents are irrelevant  → fall back to web search (or broad re-query).
  4. If some are ambiguous           → supplement with web search.
  5. If relevant documents found     → generate answer from them.
  6. Optionally decompose & rewrite the query for better retrieval.
"""
from __future__ import annotations
import json
from app.services.azure_openai import chat_completion
from app.services.vector_store import get_vector_store, SearchResult

GRADE_PROMPT = """You are a document relevance grader.
Given a user question and a retrieved document, output a JSON object:
{{"relevant": true/false, "reason": "one-sentence explanation"}}

Question: {question}
Document: {document}
"""

REWRITE_PROMPT = """You are a query rewriter. Rewrite the given question to be more specific
and search-engine friendly. Output the rewritten query as plain text only.

Original question: {question}
"""

GENERATE_PROMPT = """Answer the question using the provided documents.
If documents are insufficient, acknowledge what you know and what is missing.

Documents:
{context}

Question: {question}
"""


async def _grade_document(question: str, document: str) -> bool:
    messages = [{"role": "user", "content": GRADE_PROMPT.format(question=question, document=document)}]
    msg = await chat_completion(messages, temperature=0.0)
    try:
        data = json.loads(msg.content)
        return bool(data.get("relevant", False))
    except Exception:
        return "true" in msg.content.lower()


async def _rewrite_query(question: str) -> str:
    messages = [{"role": "user", "content": REWRITE_PROMPT.format(question=question)}]
    msg = await chat_completion(messages, temperature=0.0)
    return msg.content.strip()


async def corrective_rag(
    query: str,
    collection: str = "default",
    top_k: int = 5,
) -> dict:
    store = get_vector_store()

    # Step 1: Retrieve initial candidates
    results = await store.similarity_search(query, collection, top_k)

    # Step 2: Grade relevance for each document
    graded: list[tuple[SearchResult, bool]] = []
    for r in results:
        is_relevant = await _grade_document(query, r.content)
        graded.append((r, is_relevant))

    relevant = [r for r, ok in graded if ok]
    irrelevant_count = sum(1 for _, ok in graded if not ok)

    action = "correct"
    supplemented: list[SearchResult] = []

    # Step 3: Decide action
    if not relevant:
        # All irrelevant → rewrite and re-retrieve
        action = "rewrite_and_search"
        rewritten = await _rewrite_query(query)
        supplemented = await store.similarity_search(rewritten, collection, top_k)
        final_docs = supplemented
    elif irrelevant_count > 0:
        # Mixed → use relevant + supplement with rewritten search
        action = "supplement"
        rewritten = await _rewrite_query(query)
        supplemented = await store.similarity_search(rewritten, collection, top_k // 2)
        final_docs = relevant + supplemented
    else:
        # All relevant — proceed normally
        final_docs = relevant

    # Step 4: Generate
    context = "\n\n".join(f"[Doc {i+1}]: {r.content}" for i, r in enumerate(final_docs))
    messages = [{"role": "user", "content": GENERATE_PROMPT.format(context=context, question=query)}]
    msg = await chat_completion(messages)

    return {
        "answer": msg.content,
        "action": action,
        "relevant_count": len(relevant),
        "sources": [{"id": r.id, "content": r.content} for r in final_docs],
    }
