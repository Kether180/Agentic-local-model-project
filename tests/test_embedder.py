"""OllamaEmbedder against a stubbed transport — no Ollama, no network."""

import json
from dataclasses import dataclass

import httpx
import pytest
from core.embedding import OllamaEmbedder


@dataclass(frozen=True)
class Call:
    url: str
    model: str
    inputs: list[str]


def stub_client(captured: list[Call]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        inputs: list[str] = list(payload["input"])
        captured.append(Call(url=str(request.url), model=str(payload["model"]), inputs=inputs))
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]] * len(inputs)})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_embed_one_calls_the_embed_endpoint():
    captured: list[Call] = []
    embedder = OllamaEmbedder(client=stub_client(captured))

    assert embedder.embed("why is the sky blue?") == [0.1, 0.2, 0.3]
    assert captured[0].url.endswith("/api/embed")
    assert captured[0].model == "qwen3-embedding:0.6b"
    assert captured[0].inputs == ["why is the sky blue?"]


def test_embed_batch_is_one_round_trip():
    captured: list[Call] = []
    embedder = OllamaEmbedder(client=stub_client(captured))

    assert len(embedder.embed_batch(["a", "b", "c"])) == 3
    assert len(captured) == 1, "a batch must not fan out into one request per text"


def test_empty_batch_makes_no_request():
    captured: list[Call] = []
    assert OllamaEmbedder(client=stub_client(captured)).embed_batch([]) == []
    assert captured == []


def test_http_error_propagates():
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="model not found")

    embedder = OllamaEmbedder(client=httpx.Client(transport=httpx.MockTransport(failing)))
    with pytest.raises(httpx.HTTPStatusError):
        embedder.embed("x")
