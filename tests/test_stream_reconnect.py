from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest

from kimi_agents_python import Model

from .conftest import make_sync_client, sse_response


def _text_chunk(content: str) -> dict:
    return {
        "id": "cmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": Model.KIMI_K2_6.value,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }


def _stop_chunk() -> dict:
    return {
        "id": "cmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": Model.KIMI_K2_6.value,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class _DropStream(httpx.SyncByteStream):
    """Yields a few SSE bytes then raises mid-stream — simulates a dropped connection."""

    def __init__(self, parts: list[bytes], exc: BaseException) -> None:
        self._parts = parts
        self._exc = exc

    def __iter__(self) -> Iterator[bytes]:
        yield from self._parts
        raise self._exc

    def close(self) -> None:  # pragma: no cover - no-op
        pass


def _sse_bytes(chunk: dict) -> bytes:
    return f"data: {json.dumps(chunk)}\n\n".encode()


def test_stream_with_reconnect_resumes_after_drop() -> None:
    """First attempt drops mid-stream; second attempt completes the response."""
    attempts: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        attempts.append(body)
        if len(attempts) == 1:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_DropStream(
                    [_sse_bytes(_text_chunk("Hello "))],
                    httpx.RemoteProtocolError(
                        "peer closed connection", request=request
                    ),
                ),
            )
        return sse_response([_text_chunk("world"), _stop_chunk(), "[DONE]"])

    client = make_sync_client(handler, retries=0)
    collected: list[str] = []
    for chunk in client.chat.stream_with_reconnect(
        model=Model.KIMI_K2_6,
        messages=[{"role": "user", "content": "say hi"}],
        retry_delay=0.0,
        max_attempts=3,
    ):
        if chunk.choices and chunk.choices[0].delta.content:
            collected.append(chunk.choices[0].delta.content)
    assert "".join(collected) == "Hello world"
    assert attempts[1]["messages"][-1]["partial"] is True
    assert attempts[1]["messages"][-1]["content"] == "Hello "


def test_stream_with_reconnect_raises_when_tool_call_seen() -> None:
    tool_chunk = {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": Model.KIMI_K2_6.value,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "id": "t",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"x"'},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_DropStream(
                [_sse_bytes(tool_chunk)],
                httpx.RemoteProtocolError("dropped", request=request),
            ),
        )

    client = make_sync_client(handler, retries=0)
    with pytest.raises(httpx.RemoteProtocolError):
        for _ in client.chat.stream_with_reconnect(
            model=Model.KIMI_K2_6,
            messages=[{"role": "user", "content": "x"}],
            retry_delay=0.0,
        ):
            pass
