"""Application settings.

Env vars are `APP_`-prefixed; nested models use a `__` delimiter, so the Ollama chat URL is
`APP_CHAT__BASE_URL`.

Chat and embeddings are **separate, parallel configs**. Each is its own discriminated union with
its own `provider`, so you can run a Bedrock chat model against local Ollama embeddings, or any
other combination, without either choice constraining the other.
"""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ModelSize = Literal["small", "medium", "large"]
"""Steps ask for a size, never a model id — that is what keeps them provider-agnostic."""


class OllamaChatSettings(BaseModel):
    provider: Literal["ollama"] = "ollama"
    base_url: str = "http://localhost:11434"
    chat_models: dict[ModelSize, str] = {
        "small": "qwen3.5:0.8b",
        "medium": "qwen3.5:2b",
        "large": "qwen3.5:4b",
    }
    reasoning: bool = False
    """qwen3.5 reasons by default and will spend a thousand tokens thinking about a citation.
    Off unless a step genuinely needs it."""
    num_predict: int = 1024
    """Hard output cap, so a runaway generation fails fast instead of holding a queue slot."""
    num_ctx: int = 8192
    """Context window. Ollama's default of 4096 is filled by the ten retrieved chunks, which
    stopped answers after a couple of tokens (`done_reason: length`)."""


class BedrockChatSettings(BaseModel):
    provider: Literal["bedrock"] = "bedrock"
    region: str = "eu-central-1"
    chat_models: dict[ModelSize, str] = {
        # Inference-profile ids, verified against list_inference_profiles in eu-central-1. Haiku
        # carries its date/version suffix; the newer aliases do not. Guessing the shape fails at
        # invoke time, not at startup.
        "small": "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        "medium": "eu.anthropic.claude-sonnet-4-6",
        "large": "eu.anthropic.claude-opus-5",
    }


ChatSettings = Annotated[OllamaChatSettings | BedrockChatSettings, Field(discriminator="provider")]


class OllamaEmbeddingSettings(BaseModel):
    provider: Literal["ollama"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen3-embedding:0.6b"
    """1024 dimensions, 32K context. Must be a model `core.embedding` knows — an unknown model id
    raises rather than silently writing vectors of the wrong width."""


class BedrockEmbeddingSettings(BaseModel):
    provider: Literal["bedrock"] = "bedrock"
    region: str = "eu-central-1"
    model: str = "eu.cohere.embed-v4:0"


EmbeddingSettings = Annotated[
    OllamaEmbeddingSettings | BedrockEmbeddingSettings, Field(discriminator="provider")
]


class BlobSettings(BaseModel):
    """S3-compatible object storage. `endpoint_url` unset = real AWS S3, set = MinIO."""

    endpoint_url: str | None = None
    bucket: str = "documents"
    region: str = "eu-central-1"
    access_key: str | None = None
    secret_key: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/take_home"
    rabbit_url: str = "amqp://guest:guest@localhost:5672/"
    chat: ChatSettings = Field(default_factory=OllamaChatSettings)
    embedding: EmbeddingSettings = Field(default_factory=OllamaEmbeddingSettings)
    blob: BlobSettings = Field(default_factory=BlobSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
