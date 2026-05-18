from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel, TypeAdapter

from kimi_agents_python import (
    AVAILABLE_MODELS,
    Model,
    ParsedChatCompletion,
    PrefilledCompletion,
    StructuredParseError,
)

from .conftest import completion_body, make_async_client, make_sync_client


def test_chat_prefill_appends_partial_and_concatenates() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json=completion_body('"flask", "django", "fastapi"]'),
        )

    client = make_sync_client(handler)
    result = client.chat.prefill(
        model=Model.KIMI_K2_6,
        messages=[{"role": "user", "content": "list three frameworks"}],
        prefill="[",
    )
    assert isinstance(result, PrefilledCompletion)
    assert result.text == '["flask", "django", "fastapi"]'
    last = captured["messages"][-1]
    assert last["role"] == "assistant"
    assert last["partial"] is True
    assert last["content"] == "["


class _Summary(BaseModel):
    title: str
    bullets: list[str]


def test_chat_parse_with_basemodel() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode()))
        body = completion_body('{"title": "GC", "bullets": ["a", "b"]}')
        return httpx.Response(200, json=body)

    client = make_sync_client(handler)
    result = client.chat.parse(
        model=Model.KIMI_K2_6,
        messages=[{"role": "user", "content": "summarize"}],
        response_format=_Summary,
    )
    assert isinstance(result, ParsedChatCompletion)
    assert result.parsed == _Summary(title="GC", bullets=["a", "b"])
    rf = captured["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "_Summary"
    assert "title" in rf["json_schema"]["schema"]["properties"]


def test_chat_parse_with_type_adapter() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_body("[1, 2, 3]"))

    client = make_sync_client(handler)
    result = client.chat.parse(
        model=Model.KIMI_K2_6,
        messages=[{"role": "user", "content": "list"}],
        response_format=TypeAdapter(list[int]),
    )
    assert result.parsed == [1, 2, 3]


async def test_async_chat_prefill_and_parse() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        # response shape depends on the request: prefill response vs parse response
        if any(m.get("partial") for m in body["messages"]):
            return httpx.Response(200, json=completion_body("X"))
        return httpx.Response(200, json=completion_body('{"v": 1}'))

    client = make_async_client(handler)
    pre = await client.chat.prefill(
        model=Model.KIMI_K2_6,
        messages=[{"role": "user", "content": "go"}],
        prefill=">>",
    )
    assert pre.text == ">>X"

    class _V(BaseModel):
        v: int

    parsed = await client.chat.parse(
        model=Model.KIMI_K2_6,
        messages=[{"role": "user", "content": "go"}],
        response_format=_V,
    )
    assert parsed.parsed.v == 1


@pytest.mark.parametrize("model", list(AVAILABLE_MODELS))
def test_chat_parse_structured_output_works_for_every_model(model: Model) -> None:
    """chat.parse() builds a strict json_schema request and returns a typed
    value for every model the client ships — including thinking models, whose
    required max_tokens default is auto-injected."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode()))
        return httpx.Response(
            200, json=completion_body('{"title": "GC", "bullets": ["a", "b"]}')
        )

    client = make_sync_client(handler)
    result = client.chat.parse(
        model=model,
        messages=[{"role": "user", "content": "summarize"}],
        response_format=_Summary,
    )
    assert isinstance(result, ParsedChatCompletion)
    assert result.parsed == _Summary(title="GC", bullets=["a", "b"])
    rf = captured["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert "title" in rf["json_schema"]["schema"]["properties"]


@pytest.mark.parametrize("model", list(AVAILABLE_MODELS))
async def test_async_chat_parse_structured_output_works_for_every_model(
    model: Model,
) -> None:
    """Async counterpart to the per-model structured-output verification."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=completion_body('{"title": "GC", "bullets": ["x"]}')
        )

    client = make_async_client(handler)
    result = await client.chat.parse(
        model=model,
        messages=[{"role": "user", "content": "summarize"}],
        response_format=_Summary,
    )
    assert result.parsed == _Summary(title="GC", bullets=["x"])


def test_chat_parse_truncated_output_raises_clear_error() -> None:
    """finish_reason='length' must surface as an actionable error, not an
    opaque pydantic 'Invalid JSON: EOF'."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=completion_body('{"title": "GC", "bul', "length")
        )

    client = make_sync_client(handler)
    with pytest.raises(StructuredParseError, match="max_tokens"):
        client.chat.parse(
            model=Model.KIMI_K2_6,
            messages=[{"role": "user", "content": "summarize"}],
            response_format=_Summary,
        )


def test_chat_parse_empty_content_raises_clear_error() -> None:
    """Empty content (e.g. budget spent on reasoning) → clear error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_body("   "))

    client = make_sync_client(handler)
    with pytest.raises(StructuredParseError, match="empty content"):
        client.chat.parse(
            model=Model.KIMI_K2_6,
            messages=[{"role": "user", "content": "summarize"}],
            response_format=_Summary,
        )


def test_chat_parse_invalid_json_wraps_decode_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_body("not json at all"))

    client = make_sync_client(handler)
    with pytest.raises(StructuredParseError, match="not valid JSON") as exc:
        client.chat.parse(
            model=Model.KIMI_K2_6,
            messages=[{"role": "user", "content": "summarize"}],
            response_format=_Summary,
        )
    assert exc.value.completion is not None
    assert exc.value.__cause__ is not None


async def test_async_chat_parse_truncated_output_raises_clear_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_body('{"v": 1', "length"))

    client = make_async_client(handler)

    class _V(BaseModel):
        v: int

    with pytest.raises(StructuredParseError, match="max_tokens"):
        await client.chat.parse(
            model=Model.KIMI_K2_6,
            messages=[{"role": "user", "content": "go"}],
            response_format=_V,
        )


def test_chat_prefill_validates_disallowed_param() -> None:
    """The body builder should still reject garbage kwargs in the prefill path."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_body("hi"))

    client = make_sync_client(handler)
    with pytest.raises(TypeError):
        client.chat.prefill(
            model=Model.KIMI_K2_6,
            messages=[{"role": "user", "content": "x"}],
            prefill="[",
            not_a_real_param=1,
        )
