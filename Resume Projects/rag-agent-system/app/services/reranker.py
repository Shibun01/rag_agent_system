"""
Reranking service using FlashRank (local, fast) or Cohere (cloud).
Reranking improves retrieval precision by re-scoring candidates with a cross-encoder.
"""
from __future__ import annotations
from app.services.vector_store import SearchResult


class FlashRankReranker:
    """Local reranker — no external API needed."""

    def __init__(self, model: str = "ms-marco-MiniLM-L-12-v2"):
        from flashrank import Ranker
        self._ranker = Ranker(model_name=model)

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        from flashrank import RerankRequest
        passages = [{"id": r.id, "text": r.content, "meta": r.metadata} for r in results]
        rerank_req = RerankRequest(query=query, passages=passages)
        ranked = self._ranker.rerank(rerank_req)
        id_to_result = {r.id: r for r in results}
        reranked = [id_to_result[p["id"]] for p in ranked[:top_k] if p["id"] in id_to_result]
        return reranked


class CohereReranker:
    """Cloud reranker via Cohere API."""

    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        import cohere
        self._client = cohere.Client(api_key)
        self._model = model

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        docs = [r.content for r in results]
        response = self._client.rerank(query=query, documents=docs, model=self._model, top_n=top_k)
        return [results[r.index] for r in response.results]


def get_reranker(backend: str = "flashrank", **kwargs) -> FlashRankReranker | CohereReranker:
    if backend == "cohere":
        return CohereReranker(**kwargs)
    return FlashRankReranker()
