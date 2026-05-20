"""Before/after micro-benchmarks for the perf-async-py312 patchset.

Runs four hot paths on synthetic data so the numbers are reproducible
without burning real API calls:

    1. SSE chunk decode
       Before: ``json.loads(payload) → ChatCompletionChunk.model_validate``
       After:  ``_CHUNK_ADAPTER.validate_json(payload)`` (pydantic-core
                parses JSON in Rust, skipping the ``dict`` round-trip).

    2. Tool-loop transcript coerce
       Hypothesis: ``Message.model_construct`` should be faster than
       ``Message.model_validate`` because it skips validation on dicts we
       already own. **Result: this benchmark disproves the hypothesis.**
       In pydantic v2 ``model_validate`` is fully Rust (pydantic-core) while
       ``model_construct`` is Python-level field assignment, and on the
       assistant-with-tool-calls path the nested ``ToolCall``/``FunctionCall``
       rebuilds cost more than the validator they bypass. The
       ``_message_construct`` helper stays in the module for inspection but
       :func:`_coerce_messages` is left on ``model_validate`` — shipping
       this swap would have been a regression.

    3. Parallel tool dispatch
       Before: ``asyncio.gather(*coros)`` — siblings keep running after
                a sibling raises (resource leak under failure).
       After:  ``asyncio.TaskGroup`` — structured concurrency cancels
                siblings deterministically. Perf is on par; the win is
                correctness, surfaced here as wall time + a failure-case
                comparison that shows ``gather`` ran every coroutine to
                completion while TaskGroup cancelled them.

    4. HTTP/2 + connection-pool limits
       Out of scope for a synthetic benchmark — needs the live API. The
       config block below shows the recommended setup.

Run it:

    uv run python examples/35_benchmark_perf.py
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from typing import Any

from pydantic import TypeAdapter

from kimi_agents_python.types import (
    ChatCompletionChunk,
    FunctionCall,
    Message,
    ToolCall,
)


# Local reference impl of "skip validation on dicts we own" — kept here
# (not in the production module) because the benchmark below disproves the
# hypothesis that it's faster than ``Message.model_validate``.
_MESSAGE_FIELDS: frozenset[str] = frozenset(
    ("role", "content", "name", "partial", "tool_call_id", "tool_calls", "reasoning_content")
)


def _tool_call_construct(tc: Any) -> ToolCall:
    if isinstance(tc, ToolCall):
        return tc
    func = tc.get("function") or {}
    return ToolCall.model_construct(
        id=tc.get("id"),
        type=tc.get("type", "function"),
        function=FunctionCall.model_construct(
            name=func.get("name"),
            arguments=func.get("arguments", ""),
        ),
    )


def _message_construct(record: dict[str, Any]) -> Message:
    if not _MESSAGE_FIELDS.issuperset(record.keys()):
        return Message.model_validate(record)
    content = record.get("content")
    if isinstance(content, list):
        return Message.model_validate(record)
    tool_calls_raw = record.get("tool_calls")
    tool_calls = (
        None
        if tool_calls_raw is None
        else [_tool_call_construct(tc) for tc in tool_calls_raw]
    )
    return Message.model_construct(
        role=record.get("role"),
        content=content,
        name=record.get("name"),
        partial=record.get("partial"),
        tool_call_id=record.get("tool_call_id"),
        tool_calls=tool_calls,
        reasoning_content=record.get("reasoning_content"),
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_ns(ns: float) -> str:
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.2f} µs"
    return f"{ns:.0f} ns"


def _stats(samples_ns: list[float]) -> tuple[float, float]:
    return statistics.mean(samples_ns), (
        statistics.stdev(samples_ns) if len(samples_ns) > 1 else 0.0
    )


def _print_row(label: str, before_ns: float, after_ns: float, n_ops: int) -> None:
    speedup = before_ns / after_ns if after_ns else float("inf")
    per_op_before = before_ns / n_ops
    per_op_after = after_ns / n_ops
    print(
        f"  {label:<32} "
        f"before {_format_ns(per_op_before):<12} "
        f"after {_format_ns(per_op_after):<12} "
        f"speedup {speedup:>5.2f}x"
    )


# ---------------------------------------------------------------------------
# Bench 1: SSE chunk decode
# ---------------------------------------------------------------------------

_CHUNK_ADAPTER: TypeAdapter[ChatCompletionChunk] = TypeAdapter(ChatCompletionChunk)


def _build_sse_payloads(n: int) -> list[str]:
    """Build n realistic ``data: {...}`` payload strings."""
    base: dict[str, Any] = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": "kimi-k2.6",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Hello, how can I help you today?",
                    "reasoning_content": "the user said hi",
                },
                "finish_reason": None,
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    }
    payload = json.dumps(base)
    return [payload] * n


def bench_sse_decode(n_ops: int = 50_000, runs: int = 5) -> None:
    payloads = _build_sse_payloads(n_ops)

    def before() -> None:
        for p in payloads:
            ChatCompletionChunk.model_validate(json.loads(p))

    def after() -> None:
        for p in payloads:
            _CHUNK_ADAPTER.validate_json(p)

    print(f"\nSSE chunk decode  ({n_ops:,} chunks × {runs} runs)")
    before_samples = [_time(before) for _ in range(runs)]
    after_samples = [_time(after) for _ in range(runs)]
    b_mean, b_std = _stats(before_samples)
    a_mean, a_std = _stats(after_samples)
    print(
        f"    before: {_format_ns(b_mean)} ± {_format_ns(b_std)}    "
        f"after: {_format_ns(a_mean)} ± {_format_ns(a_std)}"
    )
    _print_row("per chunk", b_mean, a_mean, n_ops)


def _time(fn) -> float:
    t0 = time.perf_counter_ns()
    fn()
    return float(time.perf_counter_ns() - t0)


# ---------------------------------------------------------------------------
# Bench 2: transcript coerce
# ---------------------------------------------------------------------------


def _build_transcript_records() -> list[dict[str, Any]]:
    """A realistic tool-loop transcript: 8 messages, one with tool_calls."""
    return [
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "Look up the weather in Tokyo."},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "I should call the weather tool",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": '{"city": "Tokyo"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "weather",
            "content": '{"temp_c": 17, "condition": "clear"}',
        },
        {
            "role": "assistant",
            "content": "It's 17°C and clear in Tokyo right now.",
            "reasoning_content": None,
        },
        {"role": "user", "content": "Thanks. And in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "name": "weather",
            "content": '{"temp_c": 11, "condition": "rain"}',
        },
    ]


def bench_transcript_coerce(n_ops: int = 5_000, runs: int = 5) -> None:
    records = _build_transcript_records()

    def before() -> None:
        for _ in range(n_ops):
            for r in records:
                Message.model_validate(r)

    def after() -> None:
        for _ in range(n_ops):
            for r in records:
                _message_construct(r)

    n_msgs = n_ops * len(records)
    print(
        f"\nTranscript coerce  ({n_ops:,} loops × {len(records)} msgs "
        f"= {n_msgs:,} messages × {runs} runs)"
    )
    before_samples = [_time(before) for _ in range(runs)]
    after_samples = [_time(after) for _ in range(runs)]
    b_mean, b_std = _stats(before_samples)
    a_mean, a_std = _stats(after_samples)
    print(
        f"    model_validate:    {_format_ns(b_mean)} ± {_format_ns(b_std)}    "
        f"model_construct: {_format_ns(a_mean)} ± {_format_ns(a_std)}"
    )
    _print_row("per message", b_mean, a_mean, n_msgs)
    if a_mean > b_mean:
        print(
            "    ► model_construct is SLOWER here. pydantic-core's Rust\n"
            "      validator beats Python-level field assignment for this\n"
            "      shape — _coerce_messages stays on model_validate."
        )

    # Sanity: assert the constructed message is structurally equivalent
    # for an assistant-with-tool-calls record (the interesting case).
    sample = records[2]
    constructed = _message_construct(sample)
    validated = Message.model_validate(sample)
    assert constructed.role == validated.role
    assert constructed.tool_calls is not None and validated.tool_calls is not None
    assert constructed.tool_calls[0].id == validated.tool_calls[0].id
    assert (
        constructed.tool_calls[0].function.arguments
        == validated.tool_calls[0].function.arguments
    )


# ---------------------------------------------------------------------------
# Bench 3: parallel tool dispatch — perf + correctness
# ---------------------------------------------------------------------------


async def _tool(name: str, delay_s: float, results: list[str]) -> str:
    await asyncio.sleep(delay_s)
    results.append(name)
    return name


async def _tool_that_raises(delay_s: float) -> str:
    await asyncio.sleep(delay_s)
    raise RuntimeError("boom")


async def bench_parallel_dispatch(
    n_tools: int = 8, delay_s: float = 0.005, runs: int = 20
) -> None:
    print(
        f"\nParallel tool dispatch  ({n_tools} tools × {delay_s * 1000:.0f} ms "
        f"× {runs} runs)"
    )

    async def before() -> None:
        sink: list[str] = []
        await asyncio.gather(
            *[_tool(f"t{i}", delay_s, sink) for i in range(n_tools)]
        )

    async def after() -> None:
        sink: list[str] = []
        async with asyncio.TaskGroup() as tg:
            for i in range(n_tools):
                tg.create_task(_tool(f"t{i}", delay_s, sink))

    # Warm up
    await before()
    await after()

    before_samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        await before()
        before_samples.append(float(time.perf_counter_ns() - t0))

    after_samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        await after()
        after_samples.append(float(time.perf_counter_ns() - t0))

    b_mean, b_std = _stats(before_samples)
    a_mean, a_std = _stats(after_samples)
    print(
        f"    before: {_format_ns(b_mean)} ± {_format_ns(b_std)}    "
        f"after: {_format_ns(a_mean)} ± {_format_ns(a_std)}"
    )
    _print_row("per turn", b_mean, a_mean, 1)

    # ---- Correctness comparison: a sibling raises mid-batch ----
    print("\n  Failure-mode comparison (one tool raises after 5 ms):")
    survivors_gather: list[str] = []

    async def gather_failure_case() -> None:
        try:
            await asyncio.gather(
                _tool_that_raises(0.005),
                _tool("t_survivor_1", 0.020, survivors_gather),
                _tool("t_survivor_2", 0.020, survivors_gather),
            )
        except RuntimeError:
            # gather does NOT cancel siblings; they finish in the background
            # and append to ``survivors_gather`` after this except returns.
            await asyncio.sleep(0.030)

    await gather_failure_case()
    print(
        f"    gather: siblings still completed → "
        f"{len(survivors_gather)}/2 ran after the error"
    )

    survivors_tg: list[str] = []

    async def taskgroup_failure_case() -> None:
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(_tool_that_raises(0.005))
                tg.create_task(_tool("t_survivor_1", 0.020, survivors_tg))
                tg.create_task(_tool("t_survivor_2", 0.020, survivors_tg))
        except* RuntimeError:
            pass

    await taskgroup_failure_case()
    print(
        f"    TaskGroup: siblings cancelled → "
        f"{len(survivors_tg)}/2 ran after the error"
    )


# ---------------------------------------------------------------------------
# Bench 4: HTTP/2 + pool limits — config only
# ---------------------------------------------------------------------------


def bench_http2_config_only() -> None:
    print("\nHTTP/2 + pool limits  (config-only, needs live API to measure)")
    print(
        "    pip install 'kimi-agents-python[http2]'\n"
        "    from kimi_agents_python import AsyncKimiClient\n"
        "    # Defaults are already tuned (Limits(max_connections=100,\n"
        "    # max_keepalive_connections=20, keepalive_expiry=30.0)) and the\n"
        "    # client auto-detects ``h2`` to enable HTTP/2 multiplexing.\n"
        "    client = AsyncKimiClient()   # nothing else to do\n"
        "    # Override if needed:\n"
        "    client = AsyncKimiClient(\n"
        "        limits=httpx.Limits(max_connections=200,\n"
        "                            max_keepalive_connections=50),\n"
        "        http2=True,\n"
        "    )"
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print(" kimi-agents-python — perf-async-py312 benchmarks ")
    print("=" * 72)
    bench_sse_decode()
    bench_transcript_coerce()
    asyncio.run(bench_parallel_dispatch())
    bench_http2_config_only()
    print()


if __name__ == "__main__":
    main()
