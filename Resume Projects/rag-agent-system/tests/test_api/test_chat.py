from __future__ import annotations

from types import SimpleNamespace

from app.api.v1.endpoints import chat as chat_endpoint
from app.models.schemas import RAGStrategy


class _FakeCompletions:
    def __init__(self, content: str):
        self._content = content

    async def create(self, **kwargs):
        if kwargs.get("stream"):
            async def _stream():
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="stream "))])
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="reply"))])

            return _stream()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))])


class _FakeChatClient:
    def __init__(self, content: str):
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


def test_chat_uses_selected_strategy_and_persists_session(client):
    captured: dict[str, object] = {}

    async def fake_strategy(**kwargs):
        captured.update(kwargs)
        return {"answer": "grounded", "sources": [{"content": "tenant-scoped context"}]}

    original_strategy = chat_endpoint._STRATEGY_MAP[RAGStrategy.naive]
    original_client = chat_endpoint.get_azure_openai_client
    chat_endpoint._STRATEGY_MAP[RAGStrategy.naive] = fake_strategy
    chat_endpoint.get_azure_openai_client = lambda: _FakeChatClient("assistant reply")
    try:
        response = client.post(
            "/api/v1/chat/",
            headers={"X-Tenant-ID": "team1"},
            json={
                "messages": [{"role": "user", "content": "What changed?"}],
                "rag_strategy": "naive",
                "collection_name": "briefings",
            },
        )
    finally:
        chat_endpoint._STRATEGY_MAP[RAGStrategy.naive] = original_strategy
        chat_endpoint.get_azure_openai_client = original_client

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"]["content"] == "assistant reply"
    assert payload["sources"] == [{"content": "tenant-scoped context"}]
    assert payload["session_id"]
    assert captured["collection"] == "team1__briefings"
    history = chat_endpoint.build_context_messages(payload["session_id"], tenant_id="team1")
    assert history[-2:] == [
        {"role": "user", "content": "What changed?"},
        {"role": "assistant", "content": "assistant reply"},
    ]


def test_stream_chat_uses_requested_strategy(client):
    captured: dict[str, object] = {}

    async def fake_strategy(**kwargs):
        captured.update(kwargs)
        return {"answer": "grounded", "sources": [{"content": "stream context"}]}

    original_strategy = chat_endpoint._STRATEGY_MAP[RAGStrategy.graph_rag]
    original_client = chat_endpoint.get_azure_openai_client
    chat_endpoint._STRATEGY_MAP[RAGStrategy.graph_rag] = fake_strategy
    chat_endpoint.get_azure_openai_client = lambda: _FakeChatClient("unused")
    try:
        response = client.post(
            "/api/v1/chat/stream",
            headers={"X-Tenant-ID": "team2"},
            json={
                "messages": [{"role": "user", "content": "Stream this"}],
                "rag_strategy": "graph_rag",
                "collection_name": "ops",
            },
        )
    finally:
        chat_endpoint._STRATEGY_MAP[RAGStrategy.graph_rag] = original_strategy
        chat_endpoint.get_azure_openai_client = original_client

    assert response.status_code == 200
    assert "data: stream " in response.text
    assert "data: reply" in response.text
    assert captured["collection"] == "team2__ops"