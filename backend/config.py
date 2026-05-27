"""Central configuration for AuditPilot.

Reads all environment variables (OpenAI, LangSmith, Redis, ChromaDB, SEC) into a
single typed ``Settings`` object. Importing this module also exports the
``LANGCHAIN_*`` variables back into ``os.environ`` so that LangChain / LangGraph
automatically emit traces to LangSmith without any per-call wiring.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, populated from the environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- OpenAI ----
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # ---- LangSmith tracing ----
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_tracing_v2: str = Field(default="true", alias="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(default="auditpilot", alias="LANGCHAIN_PROJECT")
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGCHAIN_ENDPOINT"
    )

    # ---- Redis ----
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # ---- ChromaDB ----
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection: str = Field(default="sec_filings", alias="CHROMA_COLLECTION")

    # ---- SEC EDGAR ----
    sec_user_agent: str = Field(
        default="AuditPilot admin@auditpilot.local", alias="SEC_USER_AGENT"
    )

    # ---- Models ----
    planner_model: str = Field(default="gpt-4o", alias="PLANNER_MODEL")
    risk_model: str = Field(default="gpt-4o", alias="RISK_MODEL")
    critic_model: str = Field(default="gpt-4o-mini", alias="CRITIC_MODEL")
    memo_model: str = Field(default="gpt-4o", alias="MEMO_MODEL")
    embed_model: str = Field(default="text-embedding-3-small", alias="EMBED_MODEL")

    # ---- RAG / chunking ----
    chunk_tokens: int = Field(default=800, alias="CHUNK_TOKENS")
    chunk_overlap_tokens: int = Field(default=100, alias="CHUNK_OVERLAP_TOKENS")
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    planner_context_chunks: int = Field(default=20, alias="PLANNER_CONTEXT_CHUNKS")

    # ---- Agent loop control ----
    max_critic_iterations: int = Field(default=2, alias="MAX_CRITIC_ITERATIONS")

    # ---- Frontend ----
    backend_url: str = Field(default="http://localhost:8000", alias="BACKEND_URL")

    # ---- Six fixed risk categories (key -> human label) ----
    @property
    def risk_categories(self) -> dict[str, str]:
        return {
            "market_risk": "Market Risk",
            "credit_risk": "Credit Risk",
            "liquidity_risk": "Liquidity Risk",
            "operational_risk": "Operational Risk",
            "regulatory_legal_risk": "Regulatory/Legal Risk",
            "strategic_risk": "Strategic Risk",
        }

    def export_langchain_env(self) -> None:
        """Push LangSmith settings into os.environ so LangChain auto-traces."""
        os.environ.setdefault("LANGCHAIN_TRACING_V2", self.langchain_tracing_v2)
        os.environ.setdefault("LANGCHAIN_PROJECT", self.langchain_project)
        os.environ.setdefault("LANGCHAIN_ENDPOINT", self.langchain_endpoint)
        if self.langchain_api_key:
            os.environ.setdefault("LANGCHAIN_API_KEY", self.langchain_api_key)
        if self.openai_api_key:
            os.environ.setdefault("OPENAI_API_KEY", self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide singleton Settings instance."""
    settings = Settings()
    settings.export_langchain_env()
    return settings


# Module-level singleton for convenient imports: `from backend.config import settings`
settings = get_settings()
