from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

_ENV_FILE = Path(__file__).parent.parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    # Azure OpenAI
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2023-05-15"
    azure_openai_deployment_name: str = "gpt-5-mini"
    azure_openai_embedding_deployment: str = "text-embedding-ada-002"

    # Vector Store (ChromaDB)
    chroma_persist_dir: str = "./data/chroma"

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    app_port: int = 8000

    # RAG defaults
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_dimensions: int = 3072  # text-embedding-3-large


@lru_cache
def get_settings() -> Settings:
    return Settings()
