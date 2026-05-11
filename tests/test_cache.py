from __future__ import annotations

import json

import httpx

from kimi_agents_python import (
    CacheStats,
    ChatCompletion,
    ChatCompletionChunk,
    KimiClient,
    Model,
    PromptTokensDetails,
    Usage,
)
from tests.conftest import completion_body, make_sync_client


def _completion_with_usage(
    *, prompt: int, cached_nested: int | None = None, cached_legacy: int | None = None
) -> dict:
    body = completion_body()
    usage: dict = {"prompt_tokens": prompt, "completion_tokens": 5, "total_tokens": prompt + 5}
    if cached_legacy is not None:
        usage["cached_tokens"] = cached_legacy
    if cached_nested is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_nested}
    body["usage"] = usage
    return body


# --- type wiring ---------------------------------------------------------------


def test_usage_parses_nested_prompt_tokens_details() -> None:
    u = Usage.model_validate(
        {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_tokens_details": {"cached_tokens": 64},
        }
    )
    assert u.prompt_tokens_details is not None
    assert u.prompt_tokens_details.cached_tokens == 64


def test_usage_nested_optional_when_absent() -> None:
    u = Usage.model_validate(
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    )
    assert u.prompt_tokens_details is None


# --- per-call cached_tokens accessor ------------------------------------------


def test_chat_completion_cached_tokens_prefers_nested() -> None:
    c = ChatCompletion.model_validate(
        _completion_with_usage(prompt=100, cached_nested=64, cached_legacy=999)
    )
    assert c.cached_tokens == 64


def test_chat_completion_cached_tokens_legacy_fallback() -> None:
    c = ChatCompletion.model_validate(_completion_with_usage(prompt=50, cached_legacy=32))
    assert c.cached_tokens == 32


def test_chat_completion_cached_tokens_zero_when_absent() -> None:
    c = ChatCompletion.model_validate(_completion_with_usage(prompt=10))
    assert c.cached_tokens == 0


def test_chat_completion_cached_tokens_zero_when_usage_missing() -> None:
    payload = _completion_with_usage(prompt=1)
    del payload["usage"]
    c = ChatCompletion.model_validate(payload)
    assert c.usage is None
    assert c.cached_tokens == 0


def test_chat_completion_chunk_cached_tokens() -> None:
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": Model.KIMI_K2_6.value,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 80,
                "completion_tokens": 4,
                "total_tokens": 84,
                "prompt_tokens_details": {"cached_tokens": 50},
            },
        }
    )
    assert chunk.cached_tokens == 50


def test_chat_completion_chunk_cached_tokens_zero_when_no_usage() -> None:
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": Model.KIMI_K2_6.value,
            "choices": [{"index": 0, "delta": {"content": "Hi"}}],
        }
    )
    assert chunk.cached_tokens == 0


# --- CacheStats record() -------------------------------------------------------


def test_stats_prefers_nested_cached_tokens() -> None:
    stats = CacheStats()
    stats.record(
        Usage(
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
            cached_tokens=999,  # legacy field — should be ignored when nested present
            prompt_tokens_details=PromptTokensDetails(cached_tokens=64),
        )
    )
    assert stats.cached_tokens == 64
    assert stats.prompt_tokens == 100
    assert stats.requests == 1


def test_stats_falls_back_to_legacy_cached_tokens() -> None:
    stats = CacheStats()
    stats.record(
        Usage(
            prompt_tokens=50,
            completion_tokens=5,
            total_tokens=55,
            cached_tokens=32,
        )
    )
    assert stats.cached_tokens == 32


def test_stats_zero_cached_when_absent() -> None:
    stats = CacheStats()
    stats.record(Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12))
    assert stats.cached_tokens == 0


def test_stats_hit_ratio() -> None:
    stats = CacheStats(prompt_tokens=200, cached_tokens=80)
    assert stats.hit_ratio == 0.4


def test_stats_hit_ratio_zero_when_empty() -> None:
    assert CacheStats().hit_ratio == 0.0


def test_stats_reset() -> None:
    stats = CacheStats(requests=3, prompt_tokens=100, cached_tokens=40)
    stats.reset()
    assert stats == CacheStats()


# --- default prompt_cache_key --------------------------------------------------


def test_default_prompt_cache_key_injected() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion_body())

    http_client = httpx.Client(
        base_url="https://api.moonshot.ai/v1",
        transport=httpx.MockTransport(handler),
    )
    with KimiClient(
        api_key="test-key",
        http_client=http_client,
        retries=0,
        prompt_cache_key="session-abc",
    ) as client:
        client.chat.create(model=Model.KIMI_K2_6, messages=[{"role": "user", "content": "x"}])

    assert captured["body"]["prompt_cache_key"] == "session-abc"


def test_per_call_prompt_cache_key_overrides_default() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion_body())

    http_client = httpx.Client(
        base_url="https://api.moonshot.ai/v1",
        transport=httpx.MockTransport(handler),
    )
    with KimiClient(
        api_key="test-key",
        http_client=http_client,
        retries=0,
        prompt_cache_key="default-key",
    ) as client:
        client.chat.create(
            model=Model.KIMI_K2_6,
            messages=[{"role": "user", "content": "x"}],
            prompt_cache_key="per-call-key",
        )

    assert captured["body"]["prompt_cache_key"] == "per-call-key"


def test_no_default_means_no_key_sent() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion_body())

    with make_sync_client(handler) as client:
        client.chat.create(model=Model.KIMI_K2_6, messages=[{"role": "user", "content": "x"}])

    assert "prompt_cache_key" not in captured["body"]


# --- stats accumulation through chat() ----------------------------------------


def test_chat_records_stats() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_completion_with_usage(prompt=100, cached_nested=64)
        )

    with make_sync_client(handler) as client:
        client.chat.create(model=Model.KIMI_K2_6, messages=[{"role": "user", "content": "x"}])
        client.chat.create(model=Model.KIMI_K2_6, messages=[{"role": "user", "content": "y"}])

    assert client.cache_stats.requests == 2
    assert client.cache_stats.prompt_tokens == 200
    assert client.cache_stats.cached_tokens == 128
    assert client.cache_stats.hit_ratio == 0.64


def test_chat_records_stats_with_legacy_field() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_completion_with_usage(prompt=50, cached_legacy=25)
        )

    with make_sync_client(handler) as client:
        client.chat.create(model=Model.KIMI_K2_6, messages=[{"role": "user", "content": "x"}])

    assert client.cache_stats.cached_tokens == 25


def test_streaming_records_stats_from_final_chunk() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        chunks = [
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
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 4,
                    "total_tokens": 84,
                    "prompt_tokens_details": {"cached_tokens": 50},
                },
            },
        ]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    with make_sync_client(handler) as client:
        list(
            client.chat.create(
                model=Model.KIMI_K2_6,
                messages=[{"role": "user", "content": "x"}],
                stream=True,
                stream_options={"include_usage": True},
            )
        )

    assert client.cache_stats.requests == 1
    assert client.cache_stats.prompt_tokens == 80
    assert client.cache_stats.cached_tokens == 50


def test_failed_chat_does_not_record_stats() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500, json={"error": {"type": "server_error", "message": "boom"}}
        )

    with make_sync_client(handler) as client:
        try:
            client.chat.create(
                model=Model.KIMI_K2_6, messages=[{"role": "user", "content": "x"}]
            )
        except Exception:
            pass
    assert client.cache_stats.requests == 0


def test_clients_have_independent_stats() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion_with_usage(prompt=10, cached_nested=5))

    with make_sync_client(handler) as a, make_sync_client(handler) as b:
        a.chat.create(model=Model.KIMI_K2_6, messages=[{"role": "user", "content": "x"}])
        assert a.cache_stats.requests == 1
        assert b.cache_stats.requests == 0


# --- async --------------------------------------------------------------------


async def test_async_chat_records_stats_and_default_key() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json=_completion_with_usage(prompt=42, cached_nested=21)
        )

    http_client = httpx.AsyncClient(
        base_url="https://api.moonshot.ai/v1",
        transport=httpx.MockTransport(handler),
    )
    from kimi_agents_python import AsyncKimiClient

    async with AsyncKimiClient(
        api_key="test-key",
        http_client=http_client,
        retries=0,
        prompt_cache_key="async-sess",
    ) as client:
        await client.chat.create(
            model=Model.KIMI_K2_6, messages=[{"role": "user", "content": "x"}]
        )

    assert captured["body"]["prompt_cache_key"] == "async-sess"
    assert client.cache_stats.cached_tokens == 21
    assert client.cache_stats.requests == 1
