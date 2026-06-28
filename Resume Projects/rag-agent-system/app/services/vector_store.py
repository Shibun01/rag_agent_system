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

    def _collection(self, name: str, create: bool = False):
        if create:
            return self._client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
        return self._client.get_collection(name)

    async def add_documents(self, documents: list[dict], collection: str = "default") -> list[str]:
        embeddings = await get_embeddings_batch([d["content"] for d in documents])
        col = self._collection(collection, create=True)
        ids = [d["id"] for d in documents]
        col.add(
            ids=ids,
            embeddings=embeddings,
            documents=[d["content"] for d in documents],
            metadatas=[d.get("metadata", {}) for d in documents],
        )
        return ids

    async def list_collections(self) -> list[dict]:
        summaries = []
        for item in self._client.list_collections():
            name = item.name if hasattr(item, "name") else str(item)
            col = self._client.get_collection(name)
            data = col.get(include=["metadatas"])
            metadatas = data.get("metadatas") or []
            document_ids = {
                meta.get("document_id") or meta.get("source") or "unknown"
                for meta in metadatas
                if meta is not None
            }
            sources = sorted({meta.get("source", "unknown") for meta in metadatas if meta is not None})
            summaries.append({
                "name": name,
                "chunk_count": col.count(),
                "document_count": len(document_ids),
                "sources": sources,
            })
        return sorted(summaries, key=lambda item: item["name"])

    async def list_documents(self, collection: str = "default") -> list[dict]:
        try:
            col = self._collection(collection)
        except Exception:
            return []
        data = col.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        grouped: dict[str, dict] = {}
        for meta in metadatas:
            meta = meta or {}
            document_id = meta.get("document_id") or meta.get("source") or "unknown"
            if document_id not in grouped:
                grouped[document_id] = {
                    "document_id": document_id,
                    "source": meta.get("source", "unknown"),
                    "chunk_count": 0,
                    "metadata": meta,
                }
            grouped[document_id]["chunk_count"] += 1
        return sorted(grouped.values(), key=lambda item: item["source"])

    async def get_document_chunks(self, collection: str, document_id: str, limit: int = 100) -> list[dict]:
        try:
            col = self._collection(collection)
        except Exception:
            return []
        data = col.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
            limit=limit,
        )
        return [
            {
                "id": chunk_id,
                "content": data.get("documents", [])[index],
                "metadata": (data.get("metadatas") or [])[index] or {},
            }
            for index, chunk_id in enumerate(data.get("ids") or [])
        ]

    async def delete_document(self, collection: str, document_id: str) -> int:
        try:
            col = self._collection(collection)
        except Exception:
            return 0
        matches = col.get(where={"document_id": document_id}, include=[])
        ids = matches.get("ids") or []
        if ids:
            col.delete(ids=ids)
        return len(ids)

    async def similarity_search(
        self,
        query: str,
        collection: str = "default",
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        try:
            col = self._collection(collection)
        except Exception:
            return []
        embedding = await get_embedding(query)
        results = col.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=filters or None,
        )
        output = []
        ids = results.get("ids") or [[]]
        if not ids or not ids[0]:
            return []
        for i, doc_id in enumerate(ids[0]):
            output.append(SearchResult(
                id=doc_id,
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                score=1 - results["distances"][0][i],
            ))
        return output

    async def delete_collection(self, collection: str) -> None:
        try:
            self._client.delete_collection(collection)
        except Exception:
            return None


_store: ChromaVectorStore | None = None


def get_vector_store() -> ChromaVectorStore:
    global _store
    if _store is None:
        _store = ChromaVectorStore()
    return _store
