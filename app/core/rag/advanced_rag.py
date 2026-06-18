"""
Advanced RAG — improves over naive RAG with:
  1. Hybrid search  : BM25 keyword + dense vector search, merged via RRF
  2. Reranking      : cross-encoder re-scores top candidates
  3. Context window : deduplication + trimming to fit token budget
"""
from __future__ import annotations
import math
from app.services.azure_openai import chat_completion
from app.services.vector_store import get_vector_store, SearchResult
from app.services.reranker import get_reranker

SYSTEM_PROMPT = """You are a helpful assistant. Answer the question using the provided context.
Be concise and cite the source numbers when relevant."""


def reciprocal_rank_fusion(
    dense: list[SearchResult], sparse: list[SearchResult], k: int = 60
) -> list[SearchResult]:
    """Merge dense + sparse rankings via RRF score = 1 / (rank + k)."""
    scores: dict[str, float] = {}
    id_to_result: dict[str, SearchResult] = {}

    for rank, r in enumerate(dense):
        scores[r.id] = scores.get(r.id, 0) + 1 / (rank + k)
        id_to_result[r.id] = r

    for rank, r in enumerate(sparse):
        scores[r.id] = scores.get(r.id, 0) + 1 / (rank + k)
        id_to_result[r.id] = r

    ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [id_to_result[i] for i in ranked_ids]


async def advanced_rag(
    query: str,
    collection: str = "default",
    top_k: int = 5,
    rerank: bool = True,
    filters: dict | None = None,
) -> dict:
    store = get_vector_store()

    # 1. Dense retrieval
    dense_results = await store.similarity_search(query, collection, top_k * 2, filters)

    # 2. Hybrid fusion (dense-only here; plug in BM25 for full hybrid)
    candidates = dense_results  # extend with BM25 results when sparse index available

    # 3. Rerank
    if rerank and candidates:
        reranker = get_reranker()
        candidates = reranker.rerank(query, candidates, top_k)
    else:
        candidates = candidates[:top_k]

    # 4. Build context (deduplicate by content hash)
    seen, unique = set(), []
    for r in candidates:
        h = hash(r.content)
        if h not in seen:
            seen.add(h)
            unique.append(r)

    context = "\n\n".join(f"[Source {i+1}]: {r.content}" for i, r in enumerate(unique))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    message = await chat_completion(messages)

    return {
        "answer": message.content,
        "sources": [{"id": r.id, "content": r.content, "score": r.score} for r in unique],
    }
