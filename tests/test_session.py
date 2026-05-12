from __future__ import annotations

import json

import httpx
import pytest

from kimi_agents_python import (
    AsyncSession,
    ManualMultiTurnError,
    Message,
    Model,
    Session,
    TokenStats,
    kimi_tool,
)
from tests.conftest import make_async_client, make_sync_client, sse_response


def _completion(
    content: str | None = "ok",
    reasoning_content: str | None = None,
    tool_calls: list[dict] | None = None,
    usage: dict | None = None,
) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "c-1",
        "object": "chat.completion",
        "created": 1,
        "model": "kimi-k2.6",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": usage
        or {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }


# -- reasoning-content round-trip (the footgun fix) ----------------------------


def test_send_echoes_reasoning_content_on_next_turn() -> None:
    """The whole point of Session: reasoning_content from turn 1 must appear
    in the assistant message echoed back to the API on turn 2."""
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(
                200,
                json=_completion(
                    content="answer-1",
                    reasoning_content="thought process step 1",
                ),
            )
        return httpx.Response(200, json=_completion(content="answer-2"))

    with make_sync_client(handler) as client:
        sess = Session(client, model=Model.KIMI_K2_6)
        sess.send("first question")
        sess.send("follow-up")

    second_messages = calls[1]["messages"]
    roles = [m["role"] for m in second_messages]
    assert roles == ["user", "assistant", "user"]
    assert second_messages[1]["reasoning_content"] == "thought process step 1"
    assert second_messages[1]["content"] == "answer-1"


def test_send_omits_reasoning_content_when_absent() -> None:
    """When the model doesn't return reasoning_content, the next turn's wire
    payload should not carry a null/empty reasoning_content field either."""
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=_completion(content="reply"))

    with make_sync_client(handler) as client:
        sess = Session(client, model=Model.KIMI_K2_6)
        sess.send("hi")
        sess.send("again")

    assistant_msg = calls[1]["messages"][1]
    assert "reasoning_content" not in assistant_msg


# -- system prompt -------------------------------------------------------------


def test_system_prompt_seeded_on_construction() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_completion(content="ok"))

    with make_sync_client(handler) as client:
        sess = Session(client, model=Model.KIMI_K2_6, system="You are terse.")
        sess.send("hi")

    msgs = captured[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "You are terse."}
    assert msgs[1]["role"] == "user"


# -- defaults + per-call overrides --------------------------------------------


def test_session_defaults_forwarded_to_chat_create() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_completion(content="ok"))

    with make_sync_client(handler) as client:
        sess = Session(
            client,
            model=Model.KIMI_K2_0905_PREVIEW,
            temperature=0.2,
            max_tokens=64,
            prompt_cache_key="key-A",
        )
        sess.send("hi")

    body = captured[0]
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 64
    assert body["prompt_cache_key"] == "key-A"


def test_send_overrides_session_defaults() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_completion(content="ok"))

    with make_sync_client(handler) as client:
        sess = Session(client, model=Model.KIMI_K2_0905_PREVIEW, temperature=0.2)
        sess.send("hi", temperature=0.9)

    assert captured[0]["temperature"] == 0.9


# -- stats ---------------------------------------------------------------------


def test_cache_stats_and_usage_accumulate_per_session() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion(
                content="ok",
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "total_tokens": 105,
                    "prompt_tokens_details": {"cached_tokens": 40},
                },
            ),
        )

    with make_sync_client(handler) as client:
        sess = Session(client, model=Model.KIMI_K2_6)
        sess.send("a")
        sess.send("b")

        assert isinstance(sess.usage, TokenStats)
        assert sess.usage.requests == 2
        assert sess.usage.prompt_tokens == 200
        assert sess.usage.completion_tokens == 10
        assert sess.usage.total_tokens == 210
        assert sess.usage.cached_tokens == 80
        assert sess.cache_stats.requests == 2
        # Per-session counter is independent from the client's global one.
        assert sess.cache_stats is not client.cache_stats
        # Client also recorded the same two calls.
        assert client.cache_stats.requests == 2


# -- streaming -----------------------------------------------------------------


def _chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> dict:
    body: dict = {
        "id": "x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "kimi-k2.6",
        "choices": [{"index": 0, "delta": delta}],
    }
    if finish is not None:
        body["choices"][0]["finish_reason"] = finish
    if usage is not None:
        body["usage"] = usage
    return body


def test_stream_appends_reassembled_assistant_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(
            [
                _chunk({"role": "assistant", "content": "Hel"}),
                _chunk({"content": "lo", "reasoning_content": "step-"}),
                _chunk({"reasoning_content": "one"}),
                _chunk(
                    {},
                    finish="stop",
                    usage={
                        "prompt_tokens": 4,
                        "completion_tokens": 2,
                        "total_tokens": 6,
                    },
                ),
                "[DONE]",
            ]
        )

    with make_sync_client(handler) as client:
        sess = Session(client, model=Model.KIMI_K2_6)
        chunks = list(sess.stream("hi"))

    assert len(chunks) == 4
    assert sess.history[-1].role == "assistant"
    assert sess.history[-1].content == "Hello"
    assert sess.history[-1].reasoning_content == "step-one"
    assert sess.usage.total_tokens == 6


def test_stream_carries_assistant_message_into_next_send() -> None:
    """After streaming, history must include the reassembled assistant turn
    so the next send() echoes it back correctly."""
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        if body.get("stream"):
            return sse_response(
                [
                    _chunk({"content": "streamed"}),
                    _chunk({"reasoning_content": "rc"}, finish="stop"),
                    "[DONE]",
                ]
            )
        return httpx.Response(200, json=_completion(content="second"))

    with make_sync_client(handler) as client:
        sess = Session(client, model=Model.KIMI_K2_6)
        for _ in sess.stream("first"):
            pass
        sess.send("second")

    follow_up = captured[1]["messages"]
    assert [m["role"] for m in follow_up] == ["user", "assistant", "user"]
    assert follow_up[1]["content"] == "streamed"
    assert follow_up[1]["reasoning_content"] == "rc"


# -- tool integration ----------------------------------------------------------


def test_send_with_tools_drives_loop_and_persists_transcript() -> None:
    @kimi_tool
    def get_weather(city: str) -> dict:
        """Lookup."""
        return {"city": city, "temp": 21}

    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(
                200,
                json=_completion(
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Tokyo"}',
                            },
                        }
                    ],
                ),
            )
        return httpx.Response(200, json=_completion(content="21°C in Tokyo."))

    with make_sync_client(handler) as client:
        sess = Session(client, model=Model.KIMI_K2_6)
        final = sess.send("weather?", tools=[get_weather])

    assert final.content == "21°C in Tokyo."
    roles = [m.role for m in sess.history]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert all(isinstance(m, Message) for m in sess.history)


# -- fork ----------------------------------------------------------------------


def test_fork_is_independent_with_fresh_stats() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(content="ok"))

    with make_sync_client(handler) as client:
        sess = Session(client, model=Model.KIMI_K2_6, system="sys")
        sess.send("a")
        assert sess.usage.requests == 1

        child = sess.fork()
        assert child.usage.requests == 0
        assert [m.role for m in child.history] == [m.role for m in sess.history]

        child.send("b")
        # Parent untouched.
        assert sess.usage.requests == 1
        assert len(sess.history) == 3  # system + user + assistant
        assert len(child.history) == 5  # + user + assistant


# -- checkpoint / restore -----------------------------------------------------


def test_checkpoint_restore_roundtrip() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(content="ok"))

    with make_sync_client(handler) as client:
        sess = Session(client, model=Model.KIMI_K2_6)
        sess.send("a")
        cid = sess.checkpoint()
        snapshot = [m.model_dump() for m in sess.history]

        sess.send("b")
        assert len(sess.history) == 4

        sess.restore(cid)
        restored = [m.model_dump() for m in sess.history]
        assert restored == snapshot


def test_restore_unknown_id_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(content="ok"))

    with make_sync_client(handler) as client:
        sess = Session(client, model=Model.KIMI_K2_6)
        with pytest.raises(KeyError):
            sess.restore("nope")


# -- reset ---------------------------------------------------------------------


def test_reset_keeps_system_and_clears_stats() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(content="ok"))

    with make_sync_client(handler) as client:
        sess = Session(client, model=Model.KIMI_K2_6, system="sys")
        sess.send("a")
        sess.checkpoint()
        assert sess.usage.requests == 1

        sess.reset()
        assert len(sess.history) == 1
        assert sess.history[0].role == "system"
        assert sess.usage.requests == 0
        assert sess.checkpoints() == []


# -- async --------------------------------------------------------------------


async def test_async_session_send_echoes_reasoning_content() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(
                200, json=_completion(content="r1", reasoning_content="rc1")
            )
        return httpx.Response(200, json=_completion(content="r2"))

    async with make_async_client(handler) as client:
        sess = AsyncSession(client, model=Model.KIMI_K2_6)
        await sess.send("q1")
        await sess.send("q2")

    assert calls[1]["messages"][1]["reasoning_content"] == "rc1"


# -- enforcement: chat.create refuses manual multi-turn -----------------------


def test_chat_create_refuses_prior_assistant_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(content="ok"))

    with make_sync_client(handler) as client:
        with pytest.raises(ManualMultiTurnError, match="Session"):
            client.chat.create(
                model=Model.KIMI_K2_6,
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "again"},
                ],
            )


def test_chat_create_refuses_tool_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(content="ok"))

    with make_sync_client(handler) as client:
        with pytest.raises(ManualMultiTurnError):
            client.chat.create(
                model=Model.KIMI_K2_6,
                messages=[
                    {"role": "user", "content": "?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "x",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "x", "content": "result"},
                ],
            )


def test_chat_create_allows_partial_mode_prefill() -> None:
    """`[system?, user, assistant(partial=True)]` is the one allowed shape."""
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_completion(content="]"))

    with make_sync_client(handler) as client:
        client.chat.create(
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[
                {"role": "user", "content": "say hi"},
                {"role": "assistant", "content": "[", "partial": True},
            ],
            max_tokens=10,
        )

    assert len(captured) == 1
    assert captured[0]["messages"][-1]["partial"] is True


def test_chat_create_refuses_assistant_without_partial_flag() -> None:
    """Trailing assistant without partial=True is still multi-turn — refuse."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(content="ok"))

    with make_sync_client(handler) as client:
        with pytest.raises(ManualMultiTurnError):
            client.chat.create(
                model=Model.KIMI_K2_6,
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            )


async def test_async_chat_create_refuses_manual_multi_turn() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(content="ok"))

    async with make_async_client(handler) as client:
        with pytest.raises(ManualMultiTurnError):
            await client.chat.create(
                model=Model.KIMI_K2_6,
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "again"},
                ],
            )


def test_session_bypasses_multi_turn_gate() -> None:
    """Session is the sanctioned multi-turn path — must not trip the gate."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(content="ok"))

    with make_sync_client(handler) as client:
        sess = Session(client, model=Model.KIMI_K2_6)
        sess.send("a")
        sess.send("b")  # second send would trip the gate on raw chat.create
        assert len(sess.history) == 4


async def test_async_session_stream_reassembles_history() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(
            [
                _chunk({"content": "Hi"}),
                _chunk({"content": " there"}, finish="stop"),
                "[DONE]",
            ]
        )

    async with make_async_client(handler) as client:
        sess = AsyncSession(client, model=Model.KIMI_K2_6)
        chunks = [c async for c in sess.stream("hi")]

    assert len(chunks) == 2
    assert sess.history[-1].content == "Hi there"
