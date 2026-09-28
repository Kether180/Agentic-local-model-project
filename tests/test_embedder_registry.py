"""Provider selection and model validation.

The registry is what makes embeddings swappable independently of the chat model, so the failure
modes that matter are: an unimplemented provider, and a model whose vector width we cannot know.
"""

import pytest
from core.config import BedrockEmbeddingSettings, OllamaEmbeddingSettings, Settings
from core.embedding import EMBEDDERS, BedrockEmbedder, Embedder, OllamaEmbedder, get_embedder


def test_chat_and_embedding_are_independent():
    """The whole point: one provider choice must not constrain the other."""
    settings = Settings.model_validate(
        {"chat": {"provider": "bedrock"}, "embedding": {"provider": "ollama"}}
    )
    assert settings.chat.provider == "bedrock"
    assert settings.embedding.provider == "ollama"
    assert isinstance(settings.embedding, OllamaEmbeddingSettings)


def test_each_config_keeps_its_own_defaults():
    settings = Settings()
    assert isinstance(settings.embedding, OllamaEmbeddingSettings)
    assert settings.embedding.model == "qwen3-embedding:0.6b"
    assert settings.chat.chat_models["small"] == "qwen3.5:0.8b"


def test_bedrock_embedding_config_selects_cohere():
    settings = Settings.model_validate({"embedding": {"provider": "bedrock"}})
    assert isinstance(settings.embedding, BedrockEmbeddingSettings)
    assert settings.embedding.model == "eu.cohere.embed-v4:0"


def test_registry_covers_every_configured_provider(monkeypatch: pytest.MonkeyPatch):
    """A provider you can configure but not instantiate is a runtime trap."""
    for provider in ("ollama", "bedrock"):
        assert provider in EMBEDDERS
        monkeypatch.setattr(
            "core.embedding.settings",
            Settings.model_validate({"embedding": {"provider": provider}}),
        )
        assert isinstance(get_embedder(), Embedder)


def test_get_embedder_returns_the_configured_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("core.embedding.settings", Settings())
    assert isinstance(get_embedder(), OllamaEmbedder)


def test_unimplemented_provider_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(EMBEDDERS, "ollama", None)  # pyright: ignore[reportArgumentType]
    monkeypatch.delitem(EMBEDDERS, "ollama")
    monkeypatch.setattr("core.embedding.settings", Settings())
    with pytest.raises(ValueError, match="no Embedder implemented for provider 'ollama'"):
        get_embedder()


def test_unknown_model_raises_rather_than_guessing_width():
    with pytest.raises(ValueError, match="does not know model 'made-up:1b'"):
        OllamaEmbedder(base_url="http://localhost:11434", model="made-up:1b")

    with pytest.raises(ValueError, match="does not know model 'amazon.titan-embed-text-v2:0'"):
        BedrockEmbedder(model="amazon.titan-embed-text-v2:0", client=object())  # pyright: ignore[reportArgumentType]


def test_known_models_expose_width_and_context():
    embedder = OllamaEmbedder(base_url="http://localhost:11434", model="qwen3-embedding:0.6b")
    assert embedder.dimensions == 1024
    assert embedder.context_tokens == 32768
