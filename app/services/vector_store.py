"""
Vector Store — ChromaDB only.
"""
from __future__ import annotations
from dataclasses import dataclass

from app.config.settings import get_settings
from app.services.azure_openai import get_embedding, get_embeddings_batch

settings = get_settings()


@dataclass
class SearchResult:
    id: str
    content: str
    metadata: dict
    score: float


class ChromaVectorStore:
    def __init__(self):
        import chromadb
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

    def _collection(self, name: str):
        return self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    async def add_documents(self, documents: list[dict], collection: str = "default") -> list[str]:
        embeddings = await get_embeddings_batch([d["content"] for d in documents])
        col = self._collection(collection)
        ids = [d["id"] for d in documents]
        col.add(
            ids=ids,
            embeddings=embeddings,
            documents=[d["content"] for d in documents],
            metadatas=[d.get("metadata", {}) for d in documents],
        )
        return ids

    async def similarity_search(
        self,
        query: str,
        collection: str = "default",
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        col = self._collection(collection)
        embedding = await get_embedding(query)
        results = col.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=filters or None,
        )
        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            output.append(SearchResult(
                id=doc_id,
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                score=1 - results["distances"][0][i],
            ))
        return output

    async def delete_collection(self, collection: str) -> None:
        self._client.delete_collection(collection)


_store: ChromaVectorStore | None = None


def get_vector_store() -> ChromaVectorStore:
    global _store
    if _store is None:
        _store = ChromaVectorStore()
    return _store
