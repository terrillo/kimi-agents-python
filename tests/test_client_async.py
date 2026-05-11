from __future__ import annotations

import json

import httpx
import pytest

from kimi_agents_python import (
    AsyncKimiClient,
    ChatCompletion,
    ChatCompletionChunk,
    KimiAPIError,
    KimiAuthenticationError,
    Model,
)
from tests.conftest import completion_body, make_async_client, sse_response


async def test_async_chat_non_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json=completion_body())

    async with make_async_client(handler) as client:
        result = await client.chat.create(
            model=Model.KIMI_K2_6,
            messages=[{"role": "user", "content": "x"}],
        )
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "hello"


async def test_async_chat_streaming() -> None:
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
                    "choices": [{"index": 0, "delta": {"content": "Hi"}}],
                },
                {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "kimi-k2.6",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
                "[DONE]",
            ]
        )

    async with make_async_client(handler) as client:
        stream = await client.chat.create(
            model=Model.KIMI_K2_6,
            messages=[{"role": "user", "content": "x"}],
            stream=True,
        )
        chunks: list[ChatCompletionChunk] = []
        async for chunk in stream:
            chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "Hi"
    assert chunks[-1].choices[0].finish_reason == "stop"


async def test_async_chat_auth_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"type": "invalid_authentication_error", "message": "no"}},
        )

    async with make_async_client(handler) as client:
        with pytest.raises(KimiAuthenticationError):
            await client.chat.create(
                model=Model.KIMI_K2_6,
                messages=[{"role": "user", "content": "x"}],
            )


async def test_async_chat_streaming_raises_on_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"type": "rate_limit_reached_error", "message": "slow"}},
        )

    async with make_async_client(handler) as client:
        with pytest.raises(KimiAPIError) as exc:
            stream = await client.chat.create(
                model=Model.KIMI_K2_6,
                messages=[{"role": "user", "content": "x"}],
                stream=True,
            )
            async for _ in stream:
                pass
    assert exc.value.status_code == 429


async def test_async_list_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "kimi-k2.6", "object": "model"}]},
        )

    async with make_async_client(handler) as client:
        models = await client.models.list()
    assert models[0].id == "kimi-k2.6"


async def test_async_balance() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/users/me/balance"
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "available_balance": 7.5,
                    "voucher_balance": 2.5,
                    "cash_balance": 5.0,
                },
            },
        )

    async with make_async_client(handler) as client:
        b = await client.account.balance()
    assert b.data.available_balance == 7.5


async def test_async_estimate_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/tokenizers/estimate-token-count"
        body = json.loads(request.content)
        assert body["model"] == "kimi-k2.6"
        return httpx.Response(200, json={"data": {"total_tokens": 17}})

    async with make_async_client(handler) as client:
        e = await client.tokenizers.estimate(
            model=Model.KIMI_K2_6,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert e.data.total_tokens == 17


async def test_async_context_manager_closes_owned_transport() -> None:
    async with AsyncKimiClient(api_key="test-key") as client:
        assert client._http.is_closed is False
    assert client._http.is_closed is True


async def test_async_context_manager_does_not_close_injected_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_body())

    async with make_async_client(handler) as client:
        pass
    assert client._http.is_closed is False
    await client._http.aclose()
