from __future__ import annotations

import asyncio
import json
import time
from enum import Enum
from typing import Annotated, Literal

import httpx
import pytest
from pydantic import Field

from kimi_agents_python import (
    KimiClient,
    KimiTool,
    KimiToolLoopError,
    LoopGuards,
    Model,
    ReadOnlyStreakExceededError,
    RepeatedToolCallError,
    TokenBudgetExceededError,
    arun_tools,
    kimi_tool,
    run_tools,
)
from kimi_agents_python.tools import _arun_tools_inner, _run_tools_inner
from tests.conftest import make_async_client, make_sync_client


# -- Decorator basics -----------------------------------------------------------


def test_bare_decorator_returns_kimi_tool() -> None:
    @kimi_tool
    def get_weather(city: str) -> dict:
        """Get current weather for a city."""
        return {"city": city}

    assert isinstance(get_weather, KimiTool)
    assert get_weather.name == "get_weather"
    assert get_weather.description == "Get current weather for a city."
    assert get_weather.is_async is False
    assert get_weather.strict is True
    assert get_weather.can_parallel is True


def test_schema_required_and_optional_parameters() -> None:
    @kimi_tool
    def fn(city: str, units: str = "c") -> dict:
        """Doc."""
        return {}

    schema = fn.params_json_schema
    assert schema["type"] == "object"
    assert set(schema["properties"].keys()) == {"city", "units"}
    assert schema["required"] == ["city"]


def test_schema_uses_annotated_field_description() -> None:
    @kimi_tool
    def fn(city: Annotated[str, Field(description="City name")]) -> str:
        """Doc."""
        return city

    assert fn.params_json_schema["properties"]["city"]["description"] == "City name"


def test_schema_literal_becomes_enum() -> None:
    @kimi_tool
    def fn(units: Literal["c", "f"] = "c") -> str:
        """Doc."""
        return units

    assert fn.params_json_schema["properties"]["units"]["enum"] == ["c", "f"]


class _Color(str, Enum):
    RED = "red"
    BLUE = "blue"


def test_schema_enum_class() -> None:
    @kimi_tool
    def fn(color: _Color) -> str:
        """Doc."""
        return color.value

    blob = json.dumps(fn.params_json_schema)
    assert "red" in blob and "blue" in blob


def test_parametrised_decorator_overrides() -> None:
    @kimi_tool(name_override="weather", description_override="Lookup.", strict=False)
    def fn(city: str) -> dict:
        """Original docstring should be ignored."""
        return {"city": city}

    assert fn.name == "weather"
    assert fn.description == "Lookup."
    assert fn.strict is False


def test_use_docstring_false() -> None:
    @kimi_tool(use_docstring=False)
    def fn(x: int) -> int:
        """Should not appear."""
        return x

    assert fn.description == ""


def test_missing_annotation_raises() -> None:
    with pytest.raises(TypeError, match="missing a type annotation"):

        @kimi_tool
        def fn(city) -> str:  # type: ignore[no-untyped-def]
            """Doc."""
            return city


def test_var_args_rejected() -> None:
    with pytest.raises(TypeError, match="not supported"):

        @kimi_tool
        def fn(*args: int) -> int:
            """Doc."""
            return sum(args)


# -- to_tool_def ----------------------------------------------------------------


def test_to_tool_def_shape() -> None:
    @kimi_tool
    def fn(city: str) -> dict:
        """Lookup."""
        return {}

    td = fn.to_tool_def()
    assert td.type == "function"
    assert td.function.name == "fn"
    assert td.function.description == "Lookup."
    assert td.function.parameters["properties"]["city"]["type"] == "string"


def test_to_tool_def_omits_can_parallel_from_serialised_form() -> None:
    """can_parallel is client-side metadata only; must never reach the wire."""

    @kimi_tool(can_parallel=False)
    def fn(x: int) -> int:
        """Doc."""
        return x

    blob = fn.to_tool_def().model_dump(by_alias=True, exclude_none=True)
    assert "can_parallel" not in json.dumps(blob)


def test_request_body_to_kimi_omits_can_parallel() -> None:
    """Even when a KimiTool with can_parallel=False is passed to chat(),
    the outbound request body has no can_parallel key anywhere."""

    @kimi_tool(can_parallel=False)
    def fn(x: int) -> int:
        """Doc."""
        return x

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion_with(content="hello"))

    with make_sync_client(handler) as client:
        client.chat.create(
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[{"role": "user", "content": "?"}],
            tools=[fn],
        )

    assert "can_parallel" not in json.dumps(captured["body"])


# -- invoke / ainvoke -----------------------------------------------------------


def test_invoke_parses_json_string_arguments_and_jsonifies_return() -> None:
    @kimi_tool
    def fn(city: str) -> dict:
        """Doc."""
        return {"city": city, "temp": 21}

    out = fn.invoke('{"city":"Tokyo"}')
    assert json.loads(out) == {"city": "Tokyo", "temp": 21}


def test_invoke_accepts_dict_arguments() -> None:
    @kimi_tool
    def fn(x: int) -> int:
        """Doc."""
        return x * 2

    assert fn.invoke({"x": 3}) == "6"


def test_invoke_string_return_passthrough() -> None:
    @kimi_tool
    def fn(s: str) -> str:
        """Doc."""
        return f"hi {s}"

    assert fn.invoke('{"s":"there"}') == "hi there"


class _Opaque:
    def __str__(self) -> str:
        return "<opaque>"


def test_invoke_non_jsonable_falls_back_to_str() -> None:
    @kimi_tool
    def fn() -> _Opaque:
        """Doc."""
        return _Opaque()

    # No args needed because schema has no required fields.
    assert fn.invoke("") == "<opaque>"


def test_invoke_failure_returns_default_error_string() -> None:
    @kimi_tool
    def fn(x: int) -> int:
        """Doc."""
        raise RuntimeError("boom")

    out = fn.invoke('{"x":1}')
    assert "RuntimeError" in out and "boom" in out


def test_invoke_failure_reraises_when_handler_is_none() -> None:
    @kimi_tool(failure_error_function=None)
    def fn(x: int) -> int:
        """Doc."""
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        fn.invoke('{"x":1}')


def test_invoke_custom_failure_handler() -> None:
    @kimi_tool(failure_error_function=lambda e: "handled")
    def fn() -> None:
        """Doc."""
        raise ValueError("ignored")

    assert fn.invoke("") == "handled"


async def test_async_tool_ainvoke() -> None:
    @kimi_tool
    async def fn(city: str) -> dict:
        """Doc."""
        await asyncio.sleep(0)
        return {"city": city}

    assert fn.is_async is True
    out = await fn.ainvoke('{"city":"Paris"}')
    assert json.loads(out) == {"city": "Paris"}


async def test_async_tool_invoke_in_running_loop_raises() -> None:
    @kimi_tool
    async def fn(x: int) -> int:
        """Doc."""
        return x

    with pytest.raises(RuntimeError, match="async"):
        fn.invoke('{"x":1}')


# -- client.chat.create() integration --------------------------------------------------


def _completion_with(
    tool_calls: list[dict] | None = None,
    content: str | None = "ok",
    reasoning_content: str | None = None,
) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return {
        "id": "c-1",
        "object": "chat.completion",
        "created": 1,
        "model": "kimi-k2-0905-preview",
        "choices": [
            {"index": 0, "message": message, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_chat_accepts_kimi_tool_instance_in_tools_list() -> None:
    @kimi_tool
    def get_weather(city: str) -> dict:
        """Lookup."""
        return {"city": city}

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion_with(content="hello"))

    with make_sync_client(handler) as client:
        client.chat.create(
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[{"role": "user", "content": "?"}],
            tools=[get_weather],
        )

    tools = captured["body"]["tools"]
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "get_weather"
    assert tools[0]["function"]["parameters"]["properties"]["city"]["type"] == "string"


# -- run_tools (sync) -----------------------------------------------------------


def test_run_tools_dispatches_then_returns_final_answer() -> None:
    @kimi_tool
    def get_weather(city: str) -> dict:
        """Lookup."""
        return {"city": city, "temp": 21}

    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json=_completion_with(
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
                    content=None,
                ),
            )
        return httpx.Response(200, json=_completion_with(content="It's 21°C in Tokyo."))

    with make_sync_client(handler) as client:
        result = run_tools(
            client,
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[{"role": "user", "content": "Weather in Tokyo?"}],
            tools=[get_weather],
        )

    assert result.choices[0].message.content == "It's 21°C in Tokyo."
    assert len(calls) == 2

    # Second request must contain assistant + tool messages.
    second_messages = calls[1]["messages"]
    roles = [m["role"] for m in second_messages]
    assert roles == ["user", "assistant", "tool"]
    tool_msg = second_messages[2]
    assert tool_msg["tool_call_id"] == "call_1"
    assert json.loads(tool_msg["content"]) == {"city": "Tokyo", "temp": 21}


def test_run_tools_unknown_tool_name_passes_placeholder() -> None:
    @kimi_tool
    def get_weather(city: str) -> dict:
        """Lookup."""
        return {"city": city}

    step = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal step
        step += 1
        if step == 1:
            return httpx.Response(
                200,
                json=_completion_with(
                    tool_calls=[
                        {
                            "id": "call_x",
                            "type": "function",
                            "function": {"name": "no_such_tool", "arguments": "{}"},
                        }
                    ],
                    content=None,
                ),
            )
        # Echo so we can see what content the loop wrote.
        body = json.loads(request.content)
        tool_msg = body["messages"][-1]
        return httpx.Response(200, json=_completion_with(content=tool_msg["content"]))

    with make_sync_client(handler) as client:
        result = run_tools(
            client,
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[{"role": "user", "content": "?"}],
            tools=[get_weather],
        )
    assert "Unknown tool" in result.choices[0].message.content


def test_run_tools_raises_when_max_steps_exhausted() -> None:
    @kimi_tool
    def loopy() -> str:
        """Doc."""
        return "x"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion_with(
                tool_calls=[
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "loopy", "arguments": "{}"},
                    }
                ],
                content=None,
            ),
        )

    with make_sync_client(handler) as client:
        with pytest.raises(KimiToolLoopError, match="max_steps=2"):
            run_tools(
                client,
                model=Model.KIMI_K2_0905_PREVIEW,
                messages=[{"role": "user", "content": "?"}],
                tools=[loopy],
                max_steps=2,
            )


# -- arun_tools (async) + can_parallel dispatch --------------------------------


def _two_tool_calls_response() -> dict:
    return _completion_with(
        tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "slow_a", "arguments": "{}"},
            },
            {
                "id": "c2",
                "type": "function",
                "function": {"name": "slow_b", "arguments": "{}"},
            },
        ],
        content=None,
    )


async def test_arun_tools_parallel_dispatch_when_all_can_parallel() -> None:
    starts: list[tuple[str, float]] = []
    ends: list[tuple[str, float]] = []

    @kimi_tool
    async def slow_a() -> str:
        """Doc."""
        starts.append(("a", time.perf_counter()))
        await asyncio.sleep(0.05)
        ends.append(("a", time.perf_counter()))
        return "a-done"

    @kimi_tool
    async def slow_b() -> str:
        """Doc."""
        starts.append(("b", time.perf_counter()))
        await asyncio.sleep(0.05)
        ends.append(("b", time.perf_counter()))
        return "b-done"

    step = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal step
        step += 1
        if step == 1:
            return httpx.Response(200, json=_two_tool_calls_response())
        return httpx.Response(200, json=_completion_with(content="done"))

    async with make_async_client(handler) as client:
        await arun_tools(
            client,
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[{"role": "user", "content": "?"}],
            tools=[slow_a, slow_b],
        )

    # Both starts happen before either end → parallel.
    first_end_time = min(t for _, t in ends)
    second_start_time = max(t for _, t in starts)
    assert second_start_time < first_end_time, (
        f"expected parallel dispatch but second_start={second_start_time} "
        f">= first_end={first_end_time}"
    )


async def test_arun_tools_partitions_parallel_from_serial() -> None:
    """Parallel-capable tools run via gather; serial tool runs sequentially after."""
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}

    async def record(name: str) -> str:
        starts[name] = time.perf_counter()
        await asyncio.sleep(0.05)
        ends[name] = time.perf_counter()
        return f"{name}-done"

    @kimi_tool
    async def slow_a() -> str:
        """Doc."""
        return await record("a")

    @kimi_tool(can_parallel=False)
    async def slow_b() -> str:
        """Doc."""
        return await record("b")

    step = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal step
        step += 1
        if step == 1:
            return httpx.Response(200, json=_two_tool_calls_response())
        return httpx.Response(200, json=_completion_with(content="done"))

    async with make_async_client(handler) as client:
        await arun_tools(
            client,
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[{"role": "user", "content": "?"}],
            tools=[slow_a, slow_b],
        )

    # Strategy A: parallel bucket first (a alone), then serial bucket (b).
    assert starts["b"] >= ends["a"], (
        f"expected serial-bucket b to start after parallel-bucket a finished: "
        f"a_end={ends['a']}, b_start={starts['b']}"
    )


async def test_arun_tools_three_tool_partition_proves_parallelism() -> None:
    """Two parallel-capable tools overlap; a serial tool does not overlap them."""
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}

    async def record(name: str) -> str:
        starts[name] = time.perf_counter()
        await asyncio.sleep(0.05)
        ends[name] = time.perf_counter()
        return f"{name}-done"

    @kimi_tool
    async def p1() -> str:
        """Doc."""
        return await record("p1")

    @kimi_tool
    async def p2() -> str:
        """Doc."""
        return await record("p2")

    @kimi_tool(can_parallel=False)
    async def s1() -> str:
        """Doc."""
        return await record("s1")

    three_calls = _completion_with(
        tool_calls=[
            {"id": "c1", "type": "function",
             "function": {"name": "p1", "arguments": "{}"}},
            {"id": "c2", "type": "function",
             "function": {"name": "s1", "arguments": "{}"}},
            {"id": "c3", "type": "function",
             "function": {"name": "p2", "arguments": "{}"}},
        ],
        content=None,
    )

    step = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal step
        step += 1
        if step == 1:
            return httpx.Response(200, json=three_calls)
        return httpx.Response(200, json=_completion_with(content="done"))

    async with make_async_client(handler) as client:
        await arun_tools(
            client,
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[{"role": "user", "content": "?"}],
            tools=[p1, p2, s1],
        )

    # p1 and p2 overlap (max start < min end).
    assert max(starts["p1"], starts["p2"]) < min(ends["p1"], ends["p2"]), (
        f"expected p1 and p2 to overlap; starts={starts}, ends={ends}"
    )
    # s1 runs in the serial bucket, after both parallel tools have finished.
    assert starts["s1"] >= max(ends["p1"], ends["p2"])


def test_run_tools_preserves_reasoning_content_across_turns() -> None:
    """kimi-k2-thinking requires prior reasoning_content in the next request."""

    @kimi_tool
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    tool_call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "echo", "arguments": '{"msg":"hi"}'},
    }
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(
                200,
                json=_completion_with(
                    tool_calls=[tool_call],
                    content=None,
                    reasoning_content="Step 1: call echo with msg=hi",
                ),
            )
        return httpx.Response(200, json=_completion_with(content="done"))

    with make_sync_client(handler) as client:
        run_tools(
            client,
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[{"role": "user", "content": "?"}],
            tools=[echo],
        )

    assistant_msg = calls[1]["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["reasoning_content"] == "Step 1: call echo with msg=hi"


def test_run_tools_omits_reasoning_content_when_none() -> None:
    """Non-thinking models don't return reasoning_content — wire payload stays clean."""

    @kimi_tool
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json=_completion_with(
                    tool_calls=[
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"msg":"hi"}'},
                        }
                    ],
                    content=None,
                ),
            )
        return httpx.Response(200, json=_completion_with(content="done"))

    with make_sync_client(handler) as client:
        run_tools(
            client,
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[{"role": "user", "content": "?"}],
            tools=[echo],
        )

    assistant_msg = calls[1]["messages"][1]
    assert "reasoning_content" not in assistant_msg


# -- LoopGuards (run_tools) ----------------------------------------------------


def _tool_call_response(name: str, args: str, *, total_tokens: int = 2) -> dict:
    body = _completion_with(
        tool_calls=[
            {
                "id": f"c-{name}-{args}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        ],
        content=None,
    )
    body["usage"]["total_tokens"] = total_tokens
    return body


def test_run_tools_token_budget_raises() -> None:
    @kimi_tool
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_tool_call_response("echo", '{"msg":"hi"}', total_tokens=300),
        )

    with make_sync_client(handler) as client:
        with pytest.raises(TokenBudgetExceededError, match="500"):
            run_tools(
                client,
                model=Model.KIMI_K2_0905_PREVIEW,
                messages=[{"role": "user", "content": "?"}],
                tools=[echo],
                max_steps=10,
                guards=LoopGuards(max_tokens=500),
            )


def test_run_tools_token_budget_passes_under_limit() -> None:
    @kimi_tool
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    step = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal step
        step += 1
        if step == 1:
            return httpx.Response(
                200,
                json=_tool_call_response("echo", '{"msg":"hi"}', total_tokens=100),
            )
        return httpx.Response(200, json=_completion_with(content="done"))

    with make_sync_client(handler) as client:
        result = run_tools(
            client,
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[{"role": "user", "content": "?"}],
            tools=[echo],
            guards=LoopGuards(max_tokens=1000),
        )
    assert result.choices[0].message.content == "done"


def test_run_tools_repeated_call_raises_after_threshold() -> None:
    """Same (name, args) emitted three turns in a row → RepeatedToolCallError."""

    @kimi_tool
    def search(q: str) -> dict:
        """Search."""
        return {"q": q, "hits": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_tool_call_response("search", '{"q":"x"}')
        )

    with make_sync_client(handler) as client:
        with pytest.raises(RepeatedToolCallError, match="search"):
            run_tools(
                client,
                model=Model.KIMI_K2_0905_PREVIEW,
                messages=[{"role": "user", "content": "?"}],
                tools=[search],
                max_steps=10,
                guards=LoopGuards(repeat_threshold=3),
            )


def test_run_tools_repeat_threshold_resets_on_different_args() -> None:
    """A different argument breaks the streak; loop continues to completion."""

    @kimi_tool
    def search(q: str) -> dict:
        """Search."""
        return {"q": q}

    step = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal step
        step += 1
        if step in (1, 2):
            return httpx.Response(
                200, json=_tool_call_response("search", '{"q":"a"}')
            )
        if step == 3:
            return httpx.Response(
                200, json=_tool_call_response("search", '{"q":"b"}')
            )
        return httpx.Response(200, json=_completion_with(content="done"))

    with make_sync_client(handler) as client:
        result = run_tools(
            client,
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[{"role": "user", "content": "?"}],
            tools=[search],
            max_steps=10,
            guards=LoopGuards(repeat_threshold=3),
        )
    assert result.choices[0].message.content == "done"


def test_run_tools_repeat_threshold_normalizes_arg_whitespace() -> None:
    """Same args formatted differently still count as repeats."""

    @kimi_tool
    def search(q: str) -> dict:
        """Search."""
        return {"q": q}

    formats = ['{"q":"x"}', '{ "q": "x" }', '{"q":  "x"}']
    step = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal step
        if step < len(formats):
            body = _tool_call_response("search", formats[step])
            step += 1
            return httpx.Response(200, json=body)
        return httpx.Response(200, json=_completion_with(content="done"))

    with make_sync_client(handler) as client:
        with pytest.raises(RepeatedToolCallError):
            run_tools(
                client,
                model=Model.KIMI_K2_0905_PREVIEW,
                messages=[{"role": "user", "content": "?"}],
                tools=[search],
                max_steps=10,
                guards=LoopGuards(repeat_threshold=3),
            )


def test_run_tools_read_only_streak_raises() -> None:
    """N consecutive calls to read_only tools without a mutating call → raise."""

    @kimi_tool(read_only=True)
    def search(q: str) -> dict:
        """Search (read-only)."""
        return {"q": q}

    args_seq = ['{"q":"a"}', '{"q":"b"}', '{"q":"c"}']
    step = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal step
        if step < len(args_seq):
            body = _tool_call_response("search", args_seq[step])
            step += 1
            return httpx.Response(200, json=body)
        return httpx.Response(200, json=_completion_with(content="done"))

    with make_sync_client(handler) as client:
        with pytest.raises(ReadOnlyStreakExceededError, match="3 consecutive"):
            run_tools(
                client,
                model=Model.KIMI_K2_0905_PREVIEW,
                messages=[{"role": "user", "content": "?"}],
                tools=[search],
                max_steps=10,
                guards=LoopGuards(read_only_streak=3),
            )


def test_run_tools_read_only_streak_resets_on_mutating_call() -> None:
    @kimi_tool(read_only=True)
    def search(q: str) -> dict:
        """Search (read-only)."""
        return {"q": q}

    @kimi_tool  # default read_only=False — mutating
    def write(value: str) -> str:
        """Write."""
        return value

    step = 0
    plan = [
        ("search", '{"q":"a"}'),
        ("search", '{"q":"b"}'),
        ("write", '{"value":"x"}'),  # resets the streak
        ("search", '{"q":"c"}'),
        ("search", '{"q":"d"}'),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal step
        if step >= len(plan):
            return httpx.Response(200, json=_completion_with(content="done"))
        name, args = plan[step]
        step += 1
        return httpx.Response(200, json=_tool_call_response(name, args))

    with make_sync_client(handler) as client:
        result = run_tools(
            client,
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[{"role": "user", "content": "?"}],
            tools=[search, write],
            max_steps=10,
            guards=LoopGuards(read_only_streak=3),
        )
    assert result.choices[0].message.content == "done"




# -- compactor hook ------------------------------------------------------------


def _fetch_tool():
    @kimi_tool
    def fetch(url: str) -> str:
        """Fetch a URL."""
        return "HUGE_BODY_" + "x" * 5000

    return fetch


def _fetch_then_done_handler(calls: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(
                200,
                json=_completion_with(
                    tool_calls=[
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "fetch", "arguments": '{"url":"u"}'},
                        }
                    ],
                    content=None,
                ),
            )
        return httpx.Response(200, json=_completion_with(content="done"))

    return handler


def _drop_tool_bodies(convo: list[dict]) -> list[dict]:
    """Compactor: stub out tool-result bodies, keep structure intact."""
    out: list[dict] = []
    for m in convo:
        if m.get("role") == "tool":
            out.append({**m, "content": "[elided]"})
        else:
            out.append(m)
    return out


def test_run_tools_compactor_rewrites_payload_but_keeps_full_transcript() -> None:
    calls: list[dict] = []
    seen: list[int] = []

    def compactor(convo: list[dict]) -> list[dict]:
        seen.append(len(convo))
        return _drop_tool_bodies(convo)

    with make_sync_client(_fetch_then_done_handler(calls)) as client:
        response, transcript, _usage = _run_tools_inner(
            client,
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[{"role": "user", "content": "?"}],
            tools=[_fetch_tool()],
            max_steps=5,
            guards=None,
            compactor=compactor,
        )

    # Called once per model turn, each time with the full running convo.
    assert seen == [1, 3]
    # The huge tool body was elided on the wire (second request).
    sent_tool_msg = next(m for m in calls[1]["messages"] if m["role"] == "tool")
    assert sent_tool_msg["content"] == "[elided]"
    # But the returned transcript still holds the full, uncompacted body.
    kept_tool_msg = next(m for m in transcript if m.get("role") == "tool")
    assert kept_tool_msg["content"].startswith("HUGE_BODY_")
    assert response.choices[0].message.content == "done"


def test_run_tools_no_compactor_sends_full_transcript() -> None:
    calls: list[dict] = []
    with make_sync_client(_fetch_then_done_handler(calls)) as client:
        run_tools(
            client,
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[{"role": "user", "content": "?"}],
            tools=[_fetch_tool()],
        )
    sent_tool_msg = next(m for m in calls[1]["messages"] if m["role"] == "tool")
    assert sent_tool_msg["content"].startswith("HUGE_BODY_")


async def test_arun_tools_compactor_rewrites_payload_but_keeps_full_transcript() -> None:
    calls: list[dict] = []
    seen: list[int] = []

    def compactor(convo: list[dict]) -> list[dict]:
        seen.append(len(convo))
        return _drop_tool_bodies(convo)

    async with make_async_client(_fetch_then_done_handler(calls)) as client:
        response, transcript, _usage = await _arun_tools_inner(
            client,
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[{"role": "user", "content": "?"}],
            tools=[_fetch_tool()],
            max_steps=5,
            guards=None,
            compactor=compactor,
        )

    assert seen == [1, 3]
    sent_tool_msg = next(m for m in calls[1]["messages"] if m["role"] == "tool")
    assert sent_tool_msg["content"] == "[elided]"
    kept_tool_msg = next(m for m in transcript if m.get("role") == "tool")
    assert kept_tool_msg["content"].startswith("HUGE_BODY_")
    assert response.choices[0].message.content == "done"
