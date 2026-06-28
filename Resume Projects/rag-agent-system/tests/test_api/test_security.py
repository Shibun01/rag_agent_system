from __future__ import annotations

from app.api.dependencies import security as security_dependencies
from app.api.v1.endpoints import documents as documents_endpoint


def test_api_key_guard_blocks_requests_when_enabled(client):
    security_dependencies.settings.auth_enabled = True
    security_dependencies.settings.api_keys = "secret-key"

    unauthorized = client.get("/api/v1/rag/strategies")
    authorized = client.get("/api/v1/rag/strategies", headers={"X-API-Key": "secret-key"})

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_collection_listing_is_tenant_scoped(client):
    class FakeStore:
        async def list_collections(self):
            return [
                {"name": "team1__alpha", "chunk_count": 4, "document_count": 1, "sources": ["a.pdf"]},
                {"name": "team2__beta", "chunk_count": 6, "document_count": 2, "sources": ["b.pdf"]},
                {"name": "shared", "chunk_count": 3, "document_count": 1, "sources": ["c.pdf"]},
            ]

    original = documents_endpoint.get_vector_store
    documents_endpoint.get_vector_store = lambda: FakeStore()
    try:
        response = client.get("/api/v1/documents/collections", headers={"X-Tenant-ID": "team1"})
    finally:
        documents_endpoint.get_vector_store = original

    assert response.status_code == 200
    assert response.json() == [
        {"name": "alpha", "chunk_count": 4, "document_count": 1, "sources": ["a.pdf"]}
    ]