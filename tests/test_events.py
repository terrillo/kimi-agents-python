from __future__ import annotations

import httpx

from kimi_agents_python import (
    Done,
    Model,
    ReasoningDelta,
    Session,
    TextDelta,
    ToolCallDelta,
    UsageEvent,
    stream_events,
)
from kimi_agents_python.types import ChatCompletionChunk

from .conftest import make_sync_client, sse_response


def _chunk(**kwargs) -> dict:
    base = {
        "id": "cmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": Model.KIMI_K2_6.value,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": None,
            }
        ],
    }
    base["choices"][0].update(kwargs)
    return base


def test_stream_events_emits_typed_events() -> None:
    chunks = [
        ChatCompletionChunk.model_validate(
            _chunk(delta={"role": "assistant", "content": "Hello"})
        ),
        ChatCompletionChunk.model_validate(
            _chunk(delta={"reasoning_content": "thinking..."})
        ),
        ChatCompletionChunk.model_validate(
            _chunk(
                delta={
                    "tool_calls": [
                        {
                            "id": "tc-1",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"x":1}'},
                        }
                    ]
                }
            )
        ),
        ChatCompletionChunk.model_validate(
            {
                "id": "cmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": Model.KIMI_K2_6.value,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 5,
                    "total_tokens": 10,
                },
            }
        ),
    ]
    events = list(stream_events(iter(chunks)))
    assert isinstance(events[0], TextDelta) and events[0].content == "Hello"
    assert isinstance(events[1], ReasoningDelta)
    assert isinstance(events[2], ToolCallDelta) and events[2].name == "f"
    # The terminal chunk carries usage + finish_reason → both events emitted.
    assert any(isinstance(e, UsageEvent) for e in events)
    assert isinstance(events[-1], Done) and events[-1].finish_reason is not None


def test_session_stream_events_updates_history() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(
            [
                _chunk(delta={"role": "assistant", "content": "hi"}),
                {
                    "id": "cmpl-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": Model.KIMI_K2_6.value,
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

    client = make_sync_client(handler)
    session = Session(client, model=Model.KIMI_K2_6, system="sys")
    events = list(session.stream_events("hello"))
    assert any(isinstance(e, TextDelta) and e.content == "hi" for e in events)
    # After draining the stream, the assistant turn is in history.
    assert session.history[-1].content == "hi"
