from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.dependencies import security as security_dependencies
from app.api.v1.endpoints import chat as chat_endpoint
from app.core.memory import conversation as conversation_memory
from app.services import analytics as analytics_service


@pytest.fixture(autouse=True)
def isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    security_dependencies.settings.auth_enabled = False
    security_dependencies.settings.api_keys = ""
    security_dependencies.settings.default_tenant_id = "default"
    security_dependencies.settings.require_tenant_header = False

    chat_endpoint.settings.top_k = 5
    conversation_memory.settings.conversation_db_path = str(tmp_path / "conversations.sqlite3")
    analytics_service.settings.chroma_persist_dir = str(tmp_path / "chroma")

    yield


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)