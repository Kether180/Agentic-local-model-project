"""The wire format is the acceptance criterion, so it gets pinned here.

The compliance runner parses every response with generated zod schemas and reports a missing field
or an unrecognised event as a failure. Both are silent locally — the server happily emits them —
so these tests are the only thing standing between a refactor and a red compliance run.

No model is called: the graph is stubbed, which keeps this in the offline suite.
"""

import json
from collections.abc import Iterator
from typing import cast
from uuid import uuid4

import pytest
from core.embedding import Embedder
from sqlalchemy.orm import Session

from api.crud.chunk import ChunkHit
from api.schemas.responses import CreateResponseBody, MessageItem, OutputText, ResponseResource
from api.services import responses as service
from api.services.agent.state import ChatContext

REQUIRED_FIELDS = [
    # Verbatim from `ResponseResource.required` in https://www.openresponses.org/openapi/openapi.json.
    # Every one of these must serialize, including the ones whose value is null.
    "id",
    "object",
    "created_at",
    "completed_at",
    "status",
    "incomplete_details",
    "model",
    "previous_response_id",
    "instructions",
    "output",
    "error",
    "tools",
    "tool_choice",
    "truncation",
    "parallel_tool_calls",
    "text",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "top_logprobs",
    "temperature",
    "reasoning",
    "usage",
    "max_output_tokens",
    "max_tool_calls",
    "store",
    "background",
    "service_tier",
    "metadata",
    "safety_identifier",
    "prompt_cache_key",
]

EXPECTED_ORDER = [
    "response.created",
    "response.in_progress",
    "response.output_item.added",
    "response.content_part.added",
    "response.output_text.delta",
    "response.output_text.annotation.added",
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done",
    "response.completed",
]


def a_hit() -> ChunkHit:
    return ChunkHit(
        chunk_id=uuid4(),
        document_id=uuid4(),
        filename="kumar-etal.pdf",
        page=4,
        text="…",
        distance=0.2,
    )


def parse_frames(frames: list[str]) -> tuple[list[tuple[str, dict[str, object]]], str]:
    """Split raw SSE text into `(event_name, payload)` pairs plus the trailing sentinel."""
    events: list[tuple[str, dict[str, object]]] = []
    sentinel = ""
    for frame in frames:
        lines = [line for line in frame.strip().split("\n") if line]
        data = next(line[len("data:") :].strip() for line in lines if line.startswith("data:"))
        if data == "[DONE]":
            sentinel = data
            continue
        name = next(
            (line[len("event:") :].strip() for line in lines if line.startswith("event:")), ""
        )
        events.append((name, json.loads(data)))
    return events, sentinel


@pytest.fixture
def stubbed_graph(monkeypatch: pytest.MonkeyPatch) -> list[ChunkHit]:
    """Replace the graph with a fixed answer that cites its one source."""
    hits = [a_hit()]

    def deltas(
        body: CreateResponseBody, size: str, context: ChatContext
    ) -> Iterator[tuple[str, list[ChunkHit]]]:
        yield "Plaques form early ", hits
        yield "[S1].", hits
        yield "", hits

    monkeypatch.setattr(service, "_answer_deltas", deltas)
    return hits


def a_body(**overrides: object) -> CreateResponseBody:
    return CreateResponseBody.model_validate({"model": "small", "input": "why?", **overrides})


def run_stream(body: CreateResponseBody) -> list[str]:
    return list(service.stream(body, cast(Session, None), cast(Embedder, None)))


def test_resource_serializes_every_required_field() -> None:
    """A dropped null fails the client's parse exactly as hard as a wrong value."""
    resource = ResponseResource(
        model="qwen3.5:0.8b", output=[MessageItem(content=[OutputText(text="hi")])]
    )

    payload = resource.serialize()

    assert sorted(payload) == sorted(REQUIRED_FIELDS)
    assert payload["completed_at"] is None, "nullable fields must survive as explicit nulls"
    assert payload["error"] is None
    assert payload["object"] == "response"


def test_input_accepts_unknown_item_fields() -> None:
    """The acceptance suite sends assistant items carrying a `phase` the spec does not define."""
    body = CreateResponseBody.model_validate(
        {
            "model": "small",
            "input": [
                {"type": "message", "role": "assistant", "phase": "commentary", "content": "a"},
                {"type": "message", "role": "user", "content": "b"},
            ],
        }
    )

    assert [m.role for m in body.messages()] == ["assistant", "user"]


def test_bare_string_input_is_a_user_message() -> None:
    assert [(m.role, m.content) for m in a_body(input="hello").messages()] == [("user", "hello")]


def test_stream_follows_the_spec_item_lifecycle(stubbed_graph: list[ChunkHit]) -> None:
    events, sentinel = parse_frames(run_stream(a_body()))
    # Collapsed, because how many deltas a model emits is not part of the contract; the order the
    # phases arrive in is.
    phases = [name for i, (name, _) in enumerate(events) if i == 0 or name != events[i - 1][0]]

    assert phases == EXPECTED_ORDER
    assert sentinel == "[DONE]", "the spec terminates the stream with a literal [DONE]"


def test_stream_event_name_matches_payload_type(stubbed_graph: list[ChunkHit]) -> None:
    """The spec requires the SSE `event:` field to equal the payload's `type`."""
    events, _ = parse_frames(run_stream(a_body()))

    assert all(name == payload["type"] for name, payload in events)


def test_sequence_numbers_are_gapless(stubbed_graph: list[ChunkHit]) -> None:
    """Clients use `sequence_number` to order events and detect gaps."""
    events, _ = parse_frames(run_stream(a_body()))

    assert [payload["sequence_number"] for _, payload in events] == list(range(len(events)))


def test_citations_stream_as_annotations(stubbed_graph: list[ChunkHit]) -> None:
    """References travel on the spec's annotation event, not a channel of our own invention."""
    events, _ = parse_frames(run_stream(a_body()))
    annotations = [p for name, p in events if name == "response.output_text.annotation.added"]

    assert len(annotations) == 1
    annotation = cast(dict[str, object], annotations[0]["annotation"])
    assert annotation["type"] == "url_citation"
    assert annotation["url"] == f"/documents/{stubbed_graph[0].document_id}#page=4"


def test_terminal_snapshot_is_a_complete_resource(stubbed_graph: list[ChunkHit]) -> None:
    """`response.completed` carries the snapshot the runner validates, so it must be whole."""
    events, _ = parse_frames(run_stream(a_body()))
    completed = next(p for name, p in events if name == "response.completed")
    response = cast(dict[str, object], completed["response"])

    assert sorted(response) == sorted(REQUIRED_FIELDS)
    assert response["status"] == "completed"
    output = cast(list[dict[str, object]], response["output"])
    content = cast(list[dict[str, object]], output[0]["content"])
    assert content[0]["text"] == "Plaques form early [S1]."
    assert len(cast(list[object], content[0]["annotations"])) == 1


def test_response_id_is_stable_across_the_stream(stubbed_graph: list[ChunkHit]) -> None:
    """One turn is one response. A client correlates the snapshots by `id`, so it cannot change."""
    events, _ = parse_frames(run_stream(a_body()))
    snapshots = [cast(dict[str, object], p["response"]) for _, p in events if "response" in p]

    assert len({cast(str, s["id"]) for s in snapshots}) == 1
    assert len({cast(int, s["created_at"]) for s in snapshots}) == 1
    assert [s["status"] for s in snapshots] == ["in_progress", "in_progress", "completed"]


def test_request_settings_are_echoed(stubbed_graph: list[ChunkHit]) -> None:
    """A client reads these back to learn what the server actually applied."""
    body = a_body(temperature=0.2, store=False, metadata={"run": "7"}, service_tier="flex")
    events, _ = parse_frames(run_stream(body))
    response = cast(dict[str, object], dict(events)["response.completed"]["response"])

    assert response["temperature"] == 0.2
    assert response["store"] is False
    assert response["metadata"] == {"run": "7"}
    assert response["service_tier"] == "flex"


def test_image_parts_are_accepted_and_described(stubbed_graph: list[ChunkHit]) -> None:
    """The suite sends an image; the default model cannot read one.

    Forwarding it is a hard 400 from Ollama, so the part is turned into a note the model can
    answer around. Dropping it silently would let the model answer as though nothing was sent.
    """
    body = CreateResponseBody.model_validate(
        {
            "model": "small",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is this?"},
                        {"type": "input_image", "image_url": "data:image/png;base64,iVBOR"},
                    ],
                }
            ],
        }
    )

    content = service.to_langchain(body.messages(), None)[0].text

    assert "What is this?" in content
    assert service.IMAGE_PLACEHOLDER in content
    assert "base64" not in content, "raw image data must not reach the model"


def test_failure_becomes_a_failed_response_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model error is a state of the response. A 500 with a traceback tells a client nothing."""

    def boom(body: CreateResponseBody, size: str, context: ChatContext) -> Iterator[object]:
        raise RuntimeError("ollama is down")
        yield  # pragma: no cover  — makes this a generator, as the real one is

    monkeypatch.setattr(service, "_answer_deltas", boom)
    events, sentinel = parse_frames(run_stream(a_body()))

    assert [name for name, _ in events][-1] == "response.failed"
    assert sentinel == "[DONE]", "a failed stream still terminates properly"
    response = cast(dict[str, object], events[-1][1]["response"])
    assert sorted(response) == sorted(REQUIRED_FIELDS)
    assert response["status"] == "failed"
    error = cast(dict[str, str], response["error"])
    assert error["code"] == "model_error"
    assert "ollama is down" in error["message"]


def test_model_resolves_to_the_provider_id() -> None:
    """`model` in the response is what generated the answer, not what the client asked for."""
    from core.config import settings

    size, model_id = service.resolve_model("medium")
    assert size == "medium"
    assert model_id == settings.chat.chat_models["medium"]

    fallback_size, fallback_id = service.resolve_model("gpt-5")
    assert fallback_size == "small"
    assert fallback_id == settings.chat.chat_models["small"]
