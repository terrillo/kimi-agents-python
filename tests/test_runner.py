from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from kimi_agents_python import (
    KimiAgent,
    LoopGuards,
    Model,
    RunCancelledError,
    RunContext,
    TokenBudgetExceededError,
    handoff,
    kimi_tool,
)
from kimi_agents_python.runner import Runner
from tests.conftest import make_async_client


# -- helpers ------------------------------------------------------------------


def _completion(
    content: str | None = "done",
    tool_calls: list[dict] | None = None,
    usage: dict | None = None,
) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
        message["content"] = None
    return {
        "id": "c-1",
        "object": "chat.completion",
        "created": 1,
        "model": "kimi-k2.6",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _tool_call(name: str, args: str, call_id: str = "call_1") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


# -- Runner.run — no tools ----------------------------------------------------


@pytest.mark.asyncio
async def test_runner_no_tools_returns_runresult():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion("hello"))

    async with make_async_client(handler) as client:
        agent = KimiAgent(name="bot", model=Model.KIMI_K2_6)
        result = await Runner.run(agent, "hi", client=client)

    assert result.final_output == "hello"
    assert result.last_agent is agent
    assert isinstance(result.cost_usd, Decimal)
    assert result.usage.requests == 1
    assert result.usage.total_tokens == 8


@pytest.mark.asyncio
async def test_runner_instructions_become_system_message():
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json=_completion("ok"))

    async with make_async_client(handler) as client:
        agent = KimiAgent(
            name="bot", model=Model.KIMI_K2_6, instructions="You are precise."
        )
        await Runner.run(agent, "question", client=client)

    msgs = captured[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are precise."
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "question"


@pytest.mark.asyncio
async def test_runner_no_instructions_skips_system_message():
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json=_completion("ok"))

    async with make_async_client(handler) as client:
        agent = KimiAgent(name="bot", model=Model.KIMI_K2_6)
        await Runner.run(agent, "hello", client=client)

    roles = [m["role"] for m in captured[0]["messages"]]
    assert roles == ["user"]


# -- Runner.run — with tools --------------------------------------------------


@pytest.mark.asyncio
async def test_runner_with_tools():
    @kimi_tool
    def get_price(model: str) -> str:
        """Return price."""
        return f"${model}=cheap"

    calls: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(json.loads(req.content))
        if len(calls) == 1:
            return httpx.Response(
                200,
                json=_completion(
                    tool_calls=[_tool_call("get_price", '{"model":"k2.6"}')]
                ),
            )
        return httpx.Response(200, json=_completion("Price is cheap."))

    async with make_async_client(handler) as client:
        agent = KimiAgent(
            name="pricer", model=Model.KIMI_K2_6, tools=[get_price]
        )
        result = await Runner.run(agent, "What's the price?", client=client)

    assert result.final_output == "Price is cheap."
    assert len(calls) == 2
    # tool result must appear in second request
    roles = [m["role"] for m in calls[1]["messages"]]
    assert "tool" in roles


# -- Runner.run — model_settings passed through ------------------------------


@pytest.mark.asyncio
async def test_runner_model_settings_forwarded():
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json=_completion("ok"))

    async with make_async_client(handler) as client:
        agent = KimiAgent(
            name="bot",
            model=Model.KIMI_K2_6,
            model_settings={"thinking": {"type": "disabled"}},
        )
        await Runner.run(agent, "go", client=client)

    assert captured[0]["thinking"] == {"type": "disabled"}


# -- Runner.run — cancellation ------------------------------------------------


@pytest.mark.asyncio
async def test_runner_cancelled_before_start():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("should not reach the network")

    async with make_async_client(handler) as client:
        ctx = RunContext()
        ctx.cancel.set()
        agent = KimiAgent(name="bot", model=Model.KIMI_K2_6)
        with pytest.raises(RunCancelledError, match="bot"):
            await Runner.run(agent, "hi", client=client, context=ctx)


# -- Runner.run — guards ------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_guards_propagated():
    calls = [0]

    def handler(req: httpx.Request) -> httpx.Response:
        calls[0] += 1
        return httpx.Response(
            200,
            json=_completion(
                tool_calls=[_tool_call("noop", "{}")],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            ),
        )

    @kimi_tool
    def noop() -> str:
        """No-op."""
        return "done"

    async with make_async_client(handler) as client:
        agent = KimiAgent(
            name="bot",
            model=Model.KIMI_K2_6,
            tools=[noop],
            guards=LoopGuards(max_tokens=1),
        )
        with pytest.raises(TokenBudgetExceededError):
            await Runner.run(agent, "go", client=client)


# -- Runner.run_sync ----------------------------------------------------------


def test_runner_run_sync():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion("sync result"))

    client = make_async_client(handler)
    agent = KimiAgent(name="sync_bot", model=Model.KIMI_K2_6)
    result = Runner.run_sync(agent, "hello", client=client)
    assert result.final_output == "sync result"


# -- Runner.run — handoffs ----------------------------------------------------


@pytest.mark.asyncio
async def test_runner_handoff_via_handoffs_list():
    """Parent receives a handoff_to_sub tool call; sub-agent completes the task."""
    call_log: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        call_log.append(body)
        n = len(call_log)

        if n == 1:
            # Parent's first turn — calls the handoff tool
            return httpx.Response(
                200,
                json=_completion(
                    tool_calls=[_tool_call("handoff_to_sub", '{"task":"do the thing"}')]
                ),
            )
        if n == 2:
            # Sub-agent's only turn
            return httpx.Response(200, json=_completion("sub result"))
        # Parent resumes with tool result, returns final answer
        return httpx.Response(200, json=_completion("final from parent"))

    async with make_async_client(handler) as client:
        sub = KimiAgent(name="sub", model=Model.KIMI_K2_6)
        parent = KimiAgent(name="parent", model=Model.KIMI_K2_6, handoffs=[sub])
        result = await Runner.run(parent, "start", client=client)

    assert result.final_output == "final from parent"
    assert len(call_log) == 3
    # The parent's tool list must contain the handoff tool
    parent_tools = call_log[0].get("tools", [])
    tool_names = [t["function"]["name"] for t in parent_tools]
    assert "handoff_to_sub" in tool_names


@pytest.mark.asyncio
async def test_handoff_public_function_builds_tool():
    """handoff() returns a KimiTool with the right name."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion("ok"))

    async with make_async_client(handler) as client:
        sub = KimiAgent(name="specialist", model=Model.KIMI_K2_6)
        tool = handoff(sub, client)
        assert tool.name == "handoff_to_specialist"
        assert tool.can_parallel is False


# -- Runner.run_parallel -------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_parallel_all_succeed():
    """Two agents run concurrently; results are returned in input order."""

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        user_msg = body["messages"][-1]["content"]
        return httpx.Response(200, json=_completion(f"answer:{user_msg}"))

    async with make_async_client(handler) as client:
        a = KimiAgent(name="a", model=Model.KIMI_K2_6)
        b = KimiAgent(name="b", model=Model.KIMI_K2_6)
        results = await Runner.run_parallel(
            [(a, "alpha"), (b, "beta")], client=client
        )

    assert len(results) == 2
    assert results[0].final_output == "answer:alpha"
    assert results[1].final_output == "answer:beta"
    assert results[0].last_agent is a
    assert results[1].last_agent is b


@pytest.mark.asyncio
async def test_runner_parallel_shared_context_cancel():
    """Shared cancelled context causes all parallel agents to raise RunCancelledError."""

    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("should not reach network")

    async with make_async_client(handler) as client:
        ctx = RunContext()
        ctx.cancel.set()
        a = KimiAgent(name="a", model=Model.KIMI_K2_6)
        b = KimiAgent(name="b", model=Model.KIMI_K2_6)
        with pytest.raises(RunCancelledError):
            await Runner.run_parallel([(a, "x"), (b, "y")], client=client, context=ctx)


@pytest.mark.asyncio
async def test_runner_parallel_independent_usage():
    """Each RunResult has its own usage tracking."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion(
                "ok",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
        )

    async with make_async_client(handler) as client:
        a = KimiAgent(name="a", model=Model.KIMI_K2_6)
        b = KimiAgent(name="b", model=Model.KIMI_K2_6)
        results = await Runner.run_parallel([(a, "q1"), (b, "q2")], client=client)

    assert results[0].usage.total_tokens == 15
    assert results[1].usage.total_tokens == 15
    # Each result has independent stats objects
    assert results[0].usage is not results[1].usage


@pytest.mark.asyncio
async def test_runner_parallel_empty_runs():
    """run_parallel with an empty sequence returns an empty list without error."""

    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("should not reach network")

    async with make_async_client(handler) as client:
        results = await Runner.run_parallel([], client=client)

    assert results == []


# -- Runner.run — tools AND handoffs together --------------------------------


@pytest.mark.asyncio
async def test_runner_tools_and_handoffs():
    """An agent with both tools and handoffs works; all appear in the tool list."""
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json=_completion("done"))

    @kimi_tool
    def local_tool() -> str:
        """A local tool."""
        return "local"

    async with make_async_client(handler) as client:
        sub = KimiAgent(name="helper", model=Model.KIMI_K2_6)
        agent = KimiAgent(
            name="parent",
            model=Model.KIMI_K2_6,
            tools=[local_tool],
            handoffs=[sub],
        )
        await Runner.run(agent, "go", client=client)

    tool_names = [t["function"]["name"] for t in captured[0].get("tools", [])]
    assert "local_tool" in tool_names
    assert "handoff_to_helper" in tool_names


# -- Runner.run — result.messages in tools path ------------------------------


@pytest.mark.asyncio
async def test_runner_result_messages_in_tools_path():
    """result.messages contains the full transcript when tools are used."""
    calls: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(json.loads(req.content))
        if len(calls) == 1:
            return httpx.Response(
                200,
                json=_completion(tool_calls=[_tool_call("ping", "{}")]),
            )
        return httpx.Response(200, json=_completion("pong"))

    @kimi_tool
    def ping() -> str:
        """Ping."""
        return "pong"

    async with make_async_client(handler) as client:
        agent = KimiAgent(name="bot", model=Model.KIMI_K2_6, tools=[ping])
        result = await Runner.run(agent, "ping me", client=client)

    roles = [m.role for m in result.messages]
    assert "user" in roles
    assert "assistant" in roles
    assert "tool" in roles


# -- Runner.run — handoff name collision guard --------------------------------


@pytest.mark.asyncio
async def test_runner_handoff_name_collision_raises():
    """Duplicate names in handoffs raise ValueError before any network call."""

    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("should not reach network")

    async with make_async_client(handler) as client:
        sub1 = KimiAgent(name="worker", model=Model.KIMI_K2_6)
        sub2 = KimiAgent(name="worker", model=Model.KIMI_K2_6)
        agent = KimiAgent(name="parent", model=Model.KIMI_K2_6, handoffs=[sub1, sub2])
        with pytest.raises(ValueError, match="worker"):
            await Runner.run(agent, "go", client=client)
