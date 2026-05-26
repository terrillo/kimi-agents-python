"""Tests for the three loop-lifecycle features.

Covers:

1. :class:`RunObserver` lifecycle hooks fire in order, with the documented
   payloads, for both the sync and the async tool loop. Async hooks are
   awaited; the parallel-dispatch branch wraps each tool invocation so
   start/finished events have correct timing.

2. :class:`KimiToolLoopError` (and every subclass) carries the
   ``usage_so_far`` and ``partial_transcript`` collected up to the trip
   point. Callers no longer need to back-fill from ``client.cache_stats``.

3. ``KimiAgent(graceful_caps=True)`` converts ``max_steps`` exhaustion and
   :class:`LoopGuards` violations into a :class:`RunResult` with
   ``truncated=True`` and the partial transcript intact, instead of
   raising.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from kimi_agents_python import (
    KimiAgent,
    LoopGuards,
    Model,
    RunObserver,
    Runner,
    arun_tools,
    kimi_tool,
    run_tools,
)
from kimi_agents_python.exceptions import (
    KimiToolLoopError,
    ReadOnlyStreakExceededError,
    RepeatedToolCallError,
    TokenBudgetExceededError,
)
from tests.conftest import make_async_client, make_sync_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(
    *,
    tool_calls: list[dict] | None = None,
    content: str | None = "ok",
    total_tokens: int = 2,
) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "c-1",
        "object": "chat.completion",
        "created": 1,
        "model": "moonshot-v1-128k",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": total_tokens,
        },
    }


def _tool_call(name: str, args: str, *, call_id: str = "c-x", total_tokens: int = 2) -> dict:
    body = _completion(
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        ],
        content=None,
        total_tokens=total_tokens,
    )
    return body


def _echo_tool():
    @kimi_tool
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    return echo


def _readonly_tool():
    @kimi_tool(read_only=True)
    def look(target: str) -> str:
        """Look at something."""
        return f"saw {target}"

    return look


class _RecordingObserver(RunObserver):
    """Captures every hook call in order with its kwargs as plain dicts."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def on_turn_start(self, *, step: int, messages: list[dict[str, Any]]) -> None:
        self.events.append(("turn_start", {"step": step, "n_msgs": len(messages)}))

    def on_compaction(self, *, step: int, before: int, after: int) -> None:
        self.events.append(
            ("compaction", {"step": step, "before": before, "after": after})
        )

    def on_chat_response(
        self,
        *,
        step: int,
        response: Any,
        usage: Any,
        usage_so_far: Any,
    ) -> None:
        self.events.append(
            (
                "chat_response",
                {
                    "step": step,
                    "total_tokens": usage.total_tokens if usage else None,
                    "cumulative_total": usage_so_far.total_tokens,
                },
            )
        )

    def on_tool_call_start(
        self,
        *,
        step: int,
        call_id: str,
        tool_name: str,
        arguments: str,
    ) -> None:
        self.events.append(
            (
                "tool_start",
                {
                    "step": step,
                    "tool": tool_name,
                    "call_id": call_id,
                    "args": arguments,
                },
            )
        )

    def on_tool_call_finished(
        self,
        *,
        step: int,
        call_id: str,
        tool_name: str,
        arguments: str,
        result: str,
        duration_ms: int,
    ) -> None:
        self.events.append(
            (
                "tool_finished",
                {
                    "step": step,
                    "tool": tool_name,
                    "call_id": call_id,
                    "result": result,
                    "duration_recorded": isinstance(duration_ms, int)
                    and duration_ms >= 0,
                },
            )
        )

    def on_step_complete(self, *, step: int, usage_so_far: Any) -> None:
        self.events.append(
            ("step_complete", {"step": step, "cumulative_total": usage_so_far.total_tokens})
        )


# ---------------------------------------------------------------------------
# (1) RunObserver lifecycle
# ---------------------------------------------------------------------------


def test_observer_sync_loop_emits_full_lifecycle_in_order() -> None:
    """One tool turn + a terminal turn → ordered events with no surprises."""

    step = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal step
        step += 1
        if step == 1:
            return httpx.Response(
                200,
                json=_tool_call("echo", '{"msg":"hi"}', call_id="c-a", total_tokens=10),
            )
        return httpx.Response(200, json=_completion(content="done", total_tokens=4))

    obs = _RecordingObserver()
    with make_sync_client(handler) as client:
        run_tools(
            client,
            model=Model.MOONSHOT_V1_128K,
            messages=[{"role": "user", "content": "?"}],
            tools=[_echo_tool()],
            observer=obs,
        )

    names = [name for name, _ in obs.events]
    assert names == [
        "turn_start",
        "chat_response",
        "tool_start",
        "tool_finished",
        "step_complete",
        "turn_start",
        "chat_response",
        "step_complete",
    ]
    # cumulative tokens advance across turns
    cumulative = [
        kw["cumulative_total"] for n, kw in obs.events if n == "chat_response"
    ]
    assert cumulative == [10, 14]
    # tool_start and tool_finished carry the same call_id and tool name
    start = next(kw for n, kw in obs.events if n == "tool_start")
    finish = next(kw for n, kw in obs.events if n == "tool_finished")
    assert start["call_id"] == finish["call_id"] == "c-a"
    assert finish["result"] == "hi"
    assert finish["duration_recorded"] is True


async def test_observer_async_loop_awaits_async_hooks() -> None:
    """Async ``on_tool_call_finished`` is awaited; result is recorded after the await."""

    step = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal step
        step += 1
        if step == 1:
            return httpx.Response(
                200,
                json=_tool_call("echo", '{"msg":"hi"}', call_id="c-a"),
            )
        return httpx.Response(200, json=_completion(content="done"))

    seen: list[str] = []

    class AsyncObs(RunObserver):
        async def on_tool_call_finished(
            self, *, step, call_id, tool_name, arguments, result, duration_ms
        ) -> None:
            seen.append(f"{tool_name}:{result}")

        async def on_chat_response(
            self, *, step, response, usage, usage_so_far
        ) -> None:
            seen.append(f"chat:{step}")

    async with make_async_client(handler) as client:
        await arun_tools(
            client,
            model=Model.MOONSHOT_V1_128K,
            messages=[{"role": "user", "content": "?"}],
            tools=[_echo_tool()],
            observer=AsyncObs(),
        )

    assert seen == ["chat:0", "echo:hi", "chat:1"]


async def test_observer_parallel_tool_calls_fire_start_and_finished_per_tool() -> None:
    """Two parallel ``can_parallel=True`` tools each get their own start+finished pair."""

    step = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal step
        step += 1
        if step == 1:
            return httpx.Response(
                200,
                json=_completion(
                    tool_calls=[
                        {
                            "id": "c-a",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"msg":"a"}',
                            },
                        },
                        {
                            "id": "c-b",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"msg":"b"}',
                            },
                        },
                    ],
                    content=None,
                ),
            )
        return httpx.Response(200, json=_completion(content="done"))

    obs = _RecordingObserver()
    async with make_async_client(handler) as client:
        await arun_tools(
            client,
            model=Model.MOONSHOT_V1_128K,
            messages=[{"role": "user", "content": "?"}],
            tools=[_echo_tool()],
            observer=obs,
        )

    starts = {kw["call_id"] for n, kw in obs.events if n == "tool_start"}
    finishes = {kw["call_id"] for n, kw in obs.events if n == "tool_finished"}
    assert starts == {"c-a", "c-b"}
    assert finishes == {"c-a", "c-b"}
    # results match arguments — wrapper preserved arg→result pairing per task
    pairs = {
        kw["call_id"]: kw["result"]
        for n, kw in obs.events
        if n == "tool_finished"
    }
    assert pairs == {"c-a": "a", "c-b": "b"}


def test_observer_compaction_fires_only_when_payload_changes() -> None:
    """``on_compaction`` runs only when the compactor actually rewrites the payload."""

    step = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal step
        step += 1
        if step == 1:
            return httpx.Response(
                200, json=_tool_call("echo", '{"msg":"hi"}', call_id="c-a")
            )
        return httpx.Response(200, json=_completion(content="done"))

    def compactor(convo: list[dict]) -> list[dict]:
        # First turn: payload has length 1 (just user). Don't change anything.
        # Second turn: drop the tool result entirely — length goes from 3 to 2.
        if len(convo) >= 3 and convo[-1].get("role") == "tool":
            return convo[:-1]
        return convo

    obs = _RecordingObserver()
    with make_sync_client(handler) as client:
        run_tools(
            client,
            model=Model.MOONSHOT_V1_128K,
            messages=[{"role": "user", "content": "?"}],
            tools=[_echo_tool()],
            compactor=compactor,
            observer=obs,
        )

    compactions = [kw for n, kw in obs.events if n == "compaction"]
    assert compactions == [{"step": 1, "before": 3, "after": 2}]


# ---------------------------------------------------------------------------
# (2) Partial state on KimiToolLoopError
# ---------------------------------------------------------------------------


def test_token_budget_error_carries_usage_and_transcript() -> None:
    """TokenBudgetExceededError exposes cumulative usage + the partial transcript."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_tool_call("echo", '{"msg":"hi"}', total_tokens=600)
        )

    with make_sync_client(handler) as client:
        with pytest.raises(TokenBudgetExceededError) as ei:
            run_tools(
                client,
                model=Model.MOONSHOT_V1_128K,
                messages=[{"role": "user", "content": "?"}],
                tools=[_echo_tool()],
                max_steps=10,
                guards=LoopGuards(max_tokens=500),
            )
    exc = ei.value
    # The turn that pushed us over is included in the transcript and the usage.
    assert exc.usage_so_far.total_tokens == 600
    assert exc.usage_so_far.requests == 1
    assert [m["role"] for m in exc.partial_transcript] == ["user", "assistant"]


def test_repeat_threshold_error_carries_partial_transcript() -> None:
    """RepeatedToolCallError attaches the transcript up to the offending assistant turn."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_tool_call("echo", '{"msg":"x"}'))

    with make_sync_client(handler) as client:
        with pytest.raises(RepeatedToolCallError) as ei:
            run_tools(
                client,
                model=Model.MOONSHOT_V1_128K,
                messages=[{"role": "user", "content": "?"}],
                tools=[_echo_tool()],
                max_steps=10,
                guards=LoopGuards(repeat_threshold=3),
            )
    exc = ei.value
    # Three turns ran (the third triggers the guard before its tool dispatches).
    # Transcript: user + 3× (assistant) + 2× (tool result). Last entry is the
    # offending assistant — no tool result was appended for it.
    roles = [m["role"] for m in exc.partial_transcript]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert exc.usage_so_far.requests == 3


def test_read_only_streak_error_carries_partial_state() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_tool_call("look", '{"target":"x"}'))

    with make_sync_client(handler) as client:
        with pytest.raises(ReadOnlyStreakExceededError) as ei:
            run_tools(
                client,
                model=Model.MOONSHOT_V1_128K,
                messages=[{"role": "user", "content": "?"}],
                tools=[_readonly_tool()],
                max_steps=10,
                guards=LoopGuards(read_only_streak=3),
            )
    exc = ei.value
    # The third read-only call trips the streak before its tool runs.
    assert exc.usage_so_far.requests == 3
    assert sum(1 for m in exc.partial_transcript if m["role"] == "assistant") == 3


def test_max_steps_exhaustion_carries_partial_state() -> None:
    """Bare ``KimiToolLoopError`` from ``max_steps`` also carries the transcript and usage."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_tool_call("echo", '{"msg":"hi"}'))

    with make_sync_client(handler) as client:
        with pytest.raises(KimiToolLoopError) as ei:
            run_tools(
                client,
                model=Model.MOONSHOT_V1_128K,
                messages=[{"role": "user", "content": "?"}],
                tools=[_echo_tool()],
                max_steps=2,
            )
    exc = ei.value
    assert type(exc) is KimiToolLoopError
    assert exc.usage_so_far.requests == 2
    # user + 2 assistant + 2 tool result
    assert len(exc.partial_transcript) == 5


# ---------------------------------------------------------------------------
# (3) graceful_caps returns RunResult(truncated=True)
# ---------------------------------------------------------------------------


async def test_graceful_caps_max_steps_returns_truncated_result() -> None:
    """An agent with ``graceful_caps=True`` and an unhittable ``max_steps`` truncates instead of raising."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_tool_call("echo", '{"msg":"hi"}'))

    agent = KimiAgent(
        name="builder",
        model=Model.MOONSHOT_V1_128K,
        tools=[_echo_tool()],
        max_steps=2,
        graceful_caps=True,
    )
    async with make_async_client(handler) as client:
        result = await Runner.run(agent, "build it", client=client)
    assert result.truncated is True
    # Each turn looked like assistant+tool; with max_steps=2 we expect:
    #   system?(none here, no instructions) + user + 2×(assistant+tool) = 5
    assert sum(1 for m in result.messages if m.role == "assistant") == 2
    assert sum(1 for m in result.messages if m.role == "tool") == 2
    assert result.usage.requests == 2


async def test_graceful_caps_token_budget_returns_truncated_result() -> None:
    """Token-budget trip with ``graceful_caps=True`` returns ``RunResult(truncated=True)``."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_tool_call("echo", '{"msg":"hi"}', total_tokens=600),
        )

    agent = KimiAgent(
        name="builder",
        model=Model.MOONSHOT_V1_128K,
        tools=[_echo_tool()],
        max_steps=10,
        guards=LoopGuards(max_tokens=500),
        graceful_caps=True,
    )
    async with make_async_client(handler) as client:
        result = await Runner.run(agent, "go", client=client)
    assert result.truncated is True
    assert result.usage.total_tokens == 600


async def test_graceful_caps_off_still_raises() -> None:
    """Without ``graceful_caps`` the same agent must raise (regression guard)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_tool_call("echo", '{"msg":"hi"}'))

    agent = KimiAgent(
        name="builder",
        model=Model.MOONSHOT_V1_128K,
        tools=[_echo_tool()],
        max_steps=2,
        graceful_caps=False,
    )
    async with make_async_client(handler) as client:
        with pytest.raises(KimiToolLoopError):
            await Runner.run(agent, "go", client=client)


def test_default_observer_invokes_no_hooks() -> None:
    """An unoverridden ``RunObserver()`` is resolved to all-``None`` and stays out of the hot path.

    Regression guard against re-introducing the ``observer.on_X`` bound-
    method dispatch in the no-op case.
    """
    from kimi_agents_python.observers import _resolve_hooks

    hooks = _resolve_hooks(RunObserver())
    assert hooks.on_turn_start is None
    assert hooks.on_compaction is None
    assert hooks.on_chat_response is None
    assert hooks.on_tool_call_start is None
    assert hooks.on_tool_call_finished is None
    assert hooks.on_step_complete is None


def test_async_observer_passed_to_sync_loop_raises_typeerror() -> None:
    """A user who accidentally writes an async hook for the sync loop gets a loud error."""

    class BadObs(RunObserver):
        async def on_chat_response(
            self, *, step, response, usage, usage_so_far
        ) -> None:
            pass

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(content="done"))

    with make_sync_client(handler) as client:
        with pytest.raises(TypeError, match="async"):
            run_tools(
                client,
                model=Model.MOONSHOT_V1_128K,
                messages=[{"role": "user", "content": "?"}],
                tools=[_echo_tool()],
                observer=BadObs(),
            )


async def test_graceful_caps_clean_run_reports_truncated_false() -> None:
    """A clean run still returns ``truncated=False`` even when ``graceful_caps=True``."""

    step = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal step
        step += 1
        if step == 1:
            return httpx.Response(200, json=_tool_call("echo", '{"msg":"hi"}'))
        return httpx.Response(200, json=_completion(content="done"))

    agent = KimiAgent(
        name="builder",
        model=Model.MOONSHOT_V1_128K,
        tools=[_echo_tool()],
        max_steps=10,
        graceful_caps=True,
    )
    async with make_async_client(handler) as client:
        result = await Runner.run(agent, "go", client=client)
    assert result.truncated is False
    assert result.final_output == "done"
