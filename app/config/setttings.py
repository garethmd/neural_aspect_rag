import logging
import os
from datetime import timedelta
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(dotenv_path="./.env")


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3


class OpenAISettings(LLMSettings):
    """OpenAI-specific settings extending LLMSettings."""

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = Field(default="gpt-4o")
    embedding_model: str = Field(default="text-embedding-3-small")


class GoogleApiSettings(LLMSettings):
    """Google API-specific settings extending LLMSettings."""

    api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    default_model: str = Field(default="gemini-2.0-flash")


class DatabaseSettings(BaseModel):
    """Database connection settings."""

    service_url: str = Field(default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL"))


class LangFuseSettings(BaseModel):
    secret_key: str = Field(default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY"))
    public_key: str = Field(default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY"))
    host: str = Field(
        default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )


class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""

    table_name: str = "posts"
    embedding_dimensions: int = 1536
    time_partition_interval: timedelta = timedelta(days=7)


class FirecrawlSettings(BaseModel):
    """Settings for Firecrawl API."""

    api_key: str = Field(default_factory=lambda: os.getenv("FIRECRAWL_API_KEY"))


class CohereSettings(BaseModel):
    """Settings for Cohere API."""

    api_key: str = Field(default_factory=lambda: os.getenv("COHERE_API_KEY"))


class Settings(BaseModel):
    """Main settings class combining all sub-settings."""

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    firecrawl: FirecrawlSettings = Field(default_factory=FirecrawlSettings)
    google: GoogleApiSettings = Field(default_factory=GoogleApiSettings)
    cohere: CohereSettings = Field(default_factory=CohereSettings)


@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    settings = Settings()
    setup_logging()
    return settings
