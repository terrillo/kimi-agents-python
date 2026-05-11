from __future__ import annotations

import json

import httpx
import pytest

from kimi_agents_python import (
    ChatCompletion,
    ChatCompletionChunk,
    KimiAPIError,
    KimiAuthenticationError,
    KimiClient,
    Model,
)
from tests.conftest import completion_body, make_sync_client, sse_response


def test_chat_non_streaming_sends_correct_body() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion_body("hi back"))

    with make_sync_client(handler) as client:
        result = client.chat(
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0.3,
            max_tokens=128,
        )

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "hi back"
    assert captured["body"]["model"] == "kimi-k2-0905-preview"
    assert captured["body"]["temperature"] == 0.3
    assert captured["body"]["max_tokens"] == 128
    assert captured["body"]["messages"] == [{"role": "user", "content": "ping"}]
    assert "stream" not in captured["body"]


def test_chat_streaming_yields_chunks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        return sse_response(
            [
                {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "kimi-k2.6",
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant", "content": ""}}
                    ],
                },
                {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "kimi-k2.6",
                    "choices": [{"index": 0, "delta": {"content": "Hello"}}],
                },
                {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "kimi-k2.6",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
                "[DONE]",
            ]
        )

    with make_sync_client(handler) as client:
        chunks = list(
            client.chat(
                model=Model.KIMI_K2_6,
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
        )

    assert len(chunks) == 3
    assert all(isinstance(c, ChatCompletionChunk) for c in chunks)
    assert chunks[1].choices[0].delta.content == "Hello"
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[-1].usage is not None


def test_chat_raises_typed_error_on_401() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"type": "invalid_authentication_error", "message": "no"}},
        )

    with make_sync_client(handler) as client:
        with pytest.raises(KimiAuthenticationError) as exc:
            client.chat(model="kimi-k2.6", messages=[{"role": "user", "content": "x"}])
    assert exc.value.status_code == 401


def test_chat_streaming_raises_on_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"type": "rate_limit_reached_error", "message": "slow"}},
        )

    with make_sync_client(handler) as client:
        with pytest.raises(KimiAPIError) as exc:
            list(
                client.chat(
                    model="kimi-k2.6",
                    messages=[{"role": "user", "content": "x"}],
                    stream=True,
                )
            )
    assert exc.value.status_code == 429


def test_list_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "kimi-k2.6", "object": "model", "context_length": 256000}
                ],
            },
        )

    with make_sync_client(handler) as client:
        models = client.list_models()
    assert len(models) == 1
    assert models[0].id == "kimi-k2.6"


def test_balance() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/users/me/balance"
        return httpx.Response(
            200,
            json={
                "code": 0,
                "status": True,
                "scode": "0x0",
                "data": {
                    "available_balance": 10.0,
                    "voucher_balance": 5.0,
                    "cash_balance": 5.0,
                },
            },
        )

    with make_sync_client(handler) as client:
        b = client.balance()
    assert b.data.available_balance == 10.0


def test_estimate_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/tokenizers/estimate-token-count"
        body = json.loads(request.content)
        assert body["model"] == "kimi-k2.6"
        return httpx.Response(200, json={"data": {"total_tokens": 42}})

    with make_sync_client(handler) as client:
        e = client.estimate_tokens(
            model=Model.KIMI_K2_6,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert e.data.total_tokens == 42


def test_chat_blocks_invalid_param_for_model_before_send() -> None:
    """Spec validation runs before any HTTP request."""
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json=completion_body())

    with make_sync_client(handler) as client:
        with pytest.raises(ValueError, match="temperature is locked"):
            client.chat(
                model=Model.KIMI_K2_6,
                messages=[{"role": "user", "content": "x"}],
                temperature=0.3,
            )
    assert called is False


def test_chat_skips_validation_for_unknown_model() -> None:
    """An unknown model id must not block the request (forward-compat)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_body())

    with make_sync_client(handler) as client:
        client.chat(
            model="kimi-k2.99-future",
            messages=[{"role": "user", "content": "x"}],
            temperature=0.42,
        )


def test_context_manager_closes_owned_transport() -> None:
    with KimiClient(api_key="test-key") as client:
        assert client._http.is_closed is False
    assert client._http.is_closed is True


def test_context_manager_does_not_close_injected_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_body())

    with make_sync_client(handler) as client:
        pass
    # Injected http clients are caller-managed: close() must NOT touch them.
    assert client._http.is_closed is False
    client._http.close()
