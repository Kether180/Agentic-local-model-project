"""Embedding providers.

`Embedder` is the seam. Concrete classes register themselves by provider name, and
`get_embedder()` returns whichever one `settings.embedding.provider` names — raising for a
provider that has no implementation rather than failing later with a confusing error.

Every embedder validates its configured model against a table of models it knows. An unknown
model id raises: we cannot know its vector width, and writing vectors of the wrong width into a
fixed-width column is a corruption bug that surfaces much later.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import httpx
from types_boto3_bedrock_runtime.client import BedrockRuntimeClient

from core.config import BedrockEmbeddingSettings, OllamaEmbeddingSettings, settings


@dataclass(frozen=True)
class ModelSpec:
    """What we must know about an embedding model before using it."""

    dimensions: int
    context_tokens: int


class Embedder(ABC):
    """Text in, vectors out.

    `embed` is for a *query*, `embed_batch` for *documents*. Most models treat these identically;
    some (Cohere) take an input type that measurably changes retrieval quality, which is why the
    two are distinct methods rather than one with a count.
    """

    models: ClassVar[dict[str, ModelSpec]] = {}
    """Models this embedder supports, by id."""

    def __init__(self, model: str) -> None:
        spec = self.models.get(model)
        if spec is None:
            raise ValueError(
                f"{type(self).__name__} does not know model {model!r}; "
                f"known models: {sorted(self.models)}"
            )
        self.model = model
        self.spec = spec

    @property
    def dimensions(self) -> int:
        """Vector width. Must equal the `Vector(...)` width in domain.pg.models."""
        return self.spec.dimensions

    @property
    def context_tokens(self) -> int:
        return self.spec.context_tokens

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed one query string."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many documents, in as few round trips as the provider allows."""


class OllamaEmbedder(Embedder):
    """POSTs to `{base_url}/api/embed`, the same call as:

    curl http://localhost:11434/api/embed \
      -d '{"model": "qwen3-embedding:0.6b", "input": "Why is the sky blue?"}'

    Deliberately raw HTTP rather than the `ollama` package, so the only dependency is httpx.
    """

    models: ClassVar[dict[str, ModelSpec]] = {
        "qwen3-embedding:0.6b": ModelSpec(dimensions=1024, context_tokens=32768),
        "qwen3-embedding:4b": ModelSpec(dimensions=2560, context_tokens=40960),
        "nomic-embed-text:v1.5": ModelSpec(dimensions=768, context_tokens=2048),
    }

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        cfg = settings.embedding
        if base_url is None and isinstance(cfg, OllamaEmbeddingSettings):
            base_url = cfg.base_url
        if base_url is None:
            raise ValueError(
                "OllamaEmbedder needs a base_url when the configured provider is not ollama"
            )
        super().__init__(model or (cfg.model if isinstance(cfg, OllamaEmbeddingSettings) else ""))
        self.base_url = base_url.rstrip("/")
        # Embedding a batch on CPU is slow; the timeout is generous on purpose.
        self._client = client or httpx.Client(timeout=300.0)

    def embed(self, text: str) -> list[float]:
        return self._post([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._post(texts)

    def _post(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list):
            raise TypeError(f"{self.base_url} returned no embeddings: {response.text[:200]}")
        return [[float(value) for value in vector] for vector in embeddings]


class BedrockEmbedder(Embedder):
    """Cohere Embed v4 through `bedrock-runtime:InvokeModel`.

    Cohere distinguishes `search_document` (what you index) from `search_query` (what you ask);
    using the wrong one costs retrieval quality, so `embed` and `embed_batch` differ here.
    `output_dimension` is pinned to the spec width — v4 also offers 256/512/1536, and changing it
    would silently stop matching the database column.
    """

    models: ClassVar[dict[str, ModelSpec]] = {
        "eu.cohere.embed-v4:0": ModelSpec(dimensions=1024, context_tokens=128000)
    }

    MAX_TOKENS = 128000

    def __init__(
        self,
        model: str | None = None,
        region: str | None = None,
        client: BedrockRuntimeClient | None = None,
    ) -> None:
        cfg = settings.embedding
        bedrock = cfg if isinstance(cfg, BedrockEmbeddingSettings) else None
        super().__init__(model or (bedrock.model if bedrock else ""))
        self.region = region or (bedrock.region if bedrock else "eu-central-1")
        if client is None:
            import boto3

            client = boto3.client("bedrock-runtime", region_name=self.region)
        self._client: BedrockRuntimeClient = client

    def embed(self, text: str) -> list[float]:
        return self._invoke([text], input_type="search_query")[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._invoke(texts, input_type="search_document")

    def _invoke(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.invoke_model(
            modelId=self.model,
            body=json.dumps(
                {
                    "input_type": input_type,
                    "texts": texts,
                    "embedding_types": ["float"],
                    "output_dimension": self.dimensions,
                    "max_tokens": self.MAX_TOKENS,
                    "truncate": "RIGHT",
                }
            ),
        )
        payload = json.loads(response["body"].read())
        # v4 nests by embedding type, since one call can return several representations.
        embeddings = payload.get("embeddings", {})
        vectors = embeddings.get("float") if isinstance(embeddings, dict) else embeddings
        if not isinstance(vectors, list):
            raise TypeError(f"{self.model} returned no float embeddings: {str(payload)[:200]}")
        return [[float(value) for value in vector] for vector in vectors]


EMBEDDERS: dict[str, Callable[[], Embedder]] = {
    "ollama": OllamaEmbedder,
    "bedrock": BedrockEmbedder,
}
"""Provider name -> a factory that builds the embedder from settings.

Typed as a zero-argument callable rather than `type[Embedder]` because that is the contract
`get_embedder` relies on: every entry must be constructible with no arguments, reading whatever it
needs from config. Adding a provider means adding a class and one entry here."""


def get_embedder() -> Embedder:
    """The embedder named by `settings.embedding.provider`."""
    provider = settings.embedding.provider
    factory = EMBEDDERS.get(provider)
    if factory is None:
        raise ValueError(
            f"no Embedder implemented for provider {provider!r}; "
            f"registered providers: {sorted(EMBEDDERS)}"
        )
    return factory()
