from __future__ import annotations

import httpx
import pytest

from kimi_agents_python import (
    EngineOverloadedError,
    KimiAuthenticationError,
    KimiRateLimitError,
    Model,
    RetryConfig,
)
from kimi_agents_python._retry import (
    DEFAULT_RETRY,
    NO_RETRY,
    _compute_delay,
    retry_async,
    retry_sync,
)
from kimi_agents_python.exceptions import KimiAPIError
from tests.conftest import completion_body, make_async_client, make_sync_client


def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make retry delays instant so tests stay fast."""
    monkeypatch.setattr("kimi_agents_python._retry.time.sleep", lambda _s: None)

    async def _async_noop(_s: float) -> None:
        return None

    monkeypatch.setattr("kimi_agents_python._retry.asyncio.sleep", _async_noop)


# --- pure helpers --------------------------------------------------------------


def test_compute_delay_uses_retry_after_when_set() -> None:
    cfg = RetryConfig(initial_delay=1, max_delay=60, jitter=0)
    assert _compute_delay(1, cfg, retry_after=4.0) == 4.0


def test_compute_delay_caps_retry_after_at_max_delay() -> None:
    cfg = RetryConfig(max_delay=5, jitter=0)
    assert _compute_delay(1, cfg, retry_after=9999.0) == 5.0


def test_compute_delay_exponential_backoff_without_jitter() -> None:
    cfg = RetryConfig(initial_delay=1.0, backoff_factor=2.0, max_delay=60, jitter=0)
    assert _compute_delay(1, cfg, None) == 1.0
    assert _compute_delay(2, cfg, None) == 2.0
    assert _compute_delay(3, cfg, None) == 4.0


def test_compute_delay_caps_at_max_delay() -> None:
    cfg = RetryConfig(initial_delay=10, backoff_factor=10, max_delay=20, jitter=0)
    assert _compute_delay(5, cfg, None) == 20.0


def test_retry_sync_no_retry_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    calls = {"n": 0}

    def fn() -> int:
        calls["n"] += 1
        return 42

    assert retry_sync(NO_RETRY, fn) == 42
    assert calls["n"] == 1


def test_retry_sync_does_not_retry_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_sleep(monkeypatch)
    calls = {"n": 0}

    def fn() -> int:
        calls["n"] += 1
        raise KimiAuthenticationError("nope", status_code=401)

    with pytest.raises(KimiAuthenticationError):
        retry_sync(DEFAULT_RETRY, fn)
    assert calls["n"] == 1


# --- sync client integration ---------------------------------------------------


def test_chat_retries_on_engine_overloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "type": "engine_overloaded_error",
                        "message": "overloaded",
                    }
                },
            )
        return httpx.Response(200, json=completion_body("recovered"))

    with make_sync_client(handler, retries=3) as client:
        result = client.chat.create(
            model=Model.KIMI_K2_6,
            messages=[{"role": "user", "content": "x"}],
        )

    assert attempts["n"] == 3
    assert result.choices[0].message.content == "recovered"


def test_chat_exhausts_retries_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(
            429,
            json={
                "error": {
                    "type": "engine_overloaded_error",
                    "message": "still overloaded",
                }
            },
        )

    with make_sync_client(handler, retries=2) as client:
        with pytest.raises(EngineOverloadedError) as exc:
            client.chat.create(
                model=Model.KIMI_K2_6,
                messages=[{"role": "user", "content": "x"}],
            )

    assert attempts["n"] == 3  # 1 initial + 2 retries
    assert exc.value.status_code == 429


def test_chat_does_not_retry_4xx_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_sleep(monkeypatch)
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(
            401, json={"error": {"type": "invalid_authentication_error", "message": "no"}}
        )

    with make_sync_client(handler, retries=3) as client:
        with pytest.raises(KimiAuthenticationError):
            client.chat.create(
                model=Model.KIMI_K2_6,
                messages=[{"role": "user", "content": "x"}],
            )

    assert attempts["n"] == 1


def test_chat_retries_on_5xx_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503, text="upstream busy")
        return httpx.Response(200, json=completion_body("ok"))

    with make_sync_client(handler, retries=2) as client:
        result = client.chat.create(
            model=Model.KIMI_K2_6,
            messages=[{"role": "user", "content": "x"}],
        )
    assert attempts["n"] == 2
    assert result.choices[0].message.content == "ok"


def test_chat_honors_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        "kimi_agents_python._retry.time.sleep", lambda s: sleeps.append(s)
    )
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "7"},
                json={
                    "error": {
                        "type": "engine_overloaded_error",
                        "message": "wait",
                    }
                },
            )
        return httpx.Response(200, json=completion_body())

    with make_sync_client(handler, retries=2) as client:
        client.chat.create(model=Model.KIMI_K2_6, messages=[{"role": "user", "content": "x"}])

    assert sleeps == [7.0]


def test_streaming_retries_open_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 2:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "type": "engine_overloaded_error",
                        "message": "overloaded",
                    }
                },
            )
        return httpx.Response(
            200,
            content=(
                'data: {"id":"x","object":"chat.completion.chunk","created":1,'
                '"model":"kimi-k2.6","choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
                "data: [DONE]\n\n"
            ).encode(),
            headers={"content-type": "text/event-stream"},
        )

    with make_sync_client(handler, retries=3) as client:
        stream = client.chat.create(
            model=Model.KIMI_K2_6,
            messages=[{"role": "user", "content": "x"}],
            stream=True,
        )
        chunks = list(stream)

    assert attempts["n"] == 2
    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == "hi"


# --- async ---------------------------------------------------------------------


async def test_async_chat_retries_on_engine_overloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_sleep(monkeypatch)
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "type": "engine_overloaded_error",
                        "message": "overloaded",
                    }
                },
            )
        return httpx.Response(200, json=completion_body("recovered"))

    async with make_async_client(handler, retries=3) as client:
        result = await client.chat.create(
            model=Model.KIMI_K2_6,
            messages=[{"role": "user", "content": "x"}],
        )

    assert attempts["n"] == 3
    assert result.choices[0].message.content == "recovered"


async def test_retry_async_does_not_retry_non_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_sleep(monkeypatch)
    calls = {"n": 0}

    async def fn() -> int:
        calls["n"] += 1
        raise KimiAuthenticationError("no", status_code=401)

    with pytest.raises(KimiAuthenticationError):
        await retry_async(DEFAULT_RETRY, fn)
    assert calls["n"] == 1


# --- exception plumbing --------------------------------------------------------


def test_api_error_retry_after_parsed_from_header() -> None:
    from kimi_agents_python._http import raise_for_status

    resp = httpx.Response(
        429,
        headers={"retry-after": "12"},
        json={"error": {"type": "engine_overloaded_error", "message": "wait"}},
    )
    with pytest.raises(KimiRateLimitError) as exc:
        raise_for_status(resp)
    assert exc.value.retry_after == 12.0


def test_api_error_retry_after_none_when_absent() -> None:
    from kimi_agents_python._http import raise_for_status

    resp = httpx.Response(
        429,
        json={"error": {"type": "engine_overloaded_error", "message": "wait"}},
    )
    with pytest.raises(KimiRateLimitError) as exc:
        raise_for_status(resp)
    assert exc.value.retry_after is None


def test_api_error_retry_after_invalid_format_is_none() -> None:
    from kimi_agents_python._http import raise_for_status

    resp = httpx.Response(
        429,
        headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"},
        json={"error": {"type": "engine_overloaded_error", "message": "wait"}},
    )
    with pytest.raises(KimiRateLimitError) as exc:
        raise_for_status(resp)
    assert exc.value.retry_after is None
