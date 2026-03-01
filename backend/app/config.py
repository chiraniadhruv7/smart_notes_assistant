"""
Environment-based configuration using Pydantic BaseSettings.
All settings are loaded from environment variables or .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # ── App ──────────────────────────────────────────────
    app_name: str = "RAG Knowledge Assistant"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:3000"]

    # ── Qdrant ───────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "knowledge_base"
    qdrant_api_key: str | None = None

    # ── Redis ────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 3600  # seconds

    # ── Embeddings ───────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 64

    # ── Retrieval ────────────────────────────────────────
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rerank_top_k: int = 5
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    rrf_k: int = 60  # reciprocal rank fusion constant

    # ── Reranker ─────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Chunking ─────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── LLM ──────────────────────────────────────────────
    llm_provider: Literal["ollama", "openai"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048

    # ── Ingestion ────────────────────────────────────────
    upload_dir: str = "./data/uploads"
    max_file_size_mb: int = 50

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


settings = get_settings()
