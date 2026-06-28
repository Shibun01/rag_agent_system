from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

_ENV_FILE = Path(__file__).parent.parent.parent / ".env"


def _csv_to_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    # Azure OpenAI
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-12-01"
    azure_openai_deployment_name: str = "gpt-5-mini"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"

    # Vector Store (ChromaDB)
    chroma_persist_dir: str = "./data/chroma"

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    app_port: int = 8000
    cors_allow_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    cors_allow_methods: str = "GET,POST,PUT,PATCH,DELETE,OPTIONS"
    cors_allow_headers: str = "Authorization,Content-Type,X-API-Key,X-Tenant-ID"
    auth_enabled: bool = False
    api_keys: str = ""
    default_tenant_id: str = "default"
    require_tenant_header: bool = False
    conversation_db_path: str = "./data/conversations.sqlite3"

    # RAG defaults
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_dimensions: int = 3072  # text-embedding-3-large

    @property
    def cors_allow_origins_list(self) -> list[str]:
        origins = _csv_to_list(self.cors_allow_origins)
        return origins or ["http://localhost:3000", "http://127.0.0.1:3000"]

    @property
    def cors_allow_methods_list(self) -> list[str]:
        methods = _csv_to_list(self.cors_allow_methods)
        return methods or ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

    @property
    def cors_allow_headers_list(self) -> list[str]:
        headers = _csv_to_list(self.cors_allow_headers)
        return headers or ["Authorization", "Content-Type", "X-API-Key", "X-Tenant-ID"]

    @property
    def api_keys_list(self) -> list[str]:
        return _csv_to_list(self.api_keys)


@lru_cache
def get_settings() -> Settings:
    return Settings()
