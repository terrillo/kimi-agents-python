"""Typed stream events.

Adapts a :class:`ChatCompletionChunk` iterator into a friendlier event stream.
Each non-empty delta field becomes its own event, the terminal usage chunk
becomes :class:`UsageEvent`, and any non-null ``finish_reason`` is emitted as
:class:`Done` at the end.

Mirrors the OpenAI Python SDK's high-level event model so UI integration
code looks the same on both providers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

from ._enums import FinishReason
from .types import ChatCompletionChunk, Usage


@dataclass(slots=True, frozen=True)
class TextDelta:
    content: str


@dataclass(slots=True, frozen=True)
class ReasoningDelta:
    content: str


@dataclass(slots=True, frozen=True)
class ToolCallDelta:
    """One incremental update to a tool call.

    Fields are exactly as they arrive on the wire — assemble across deltas by
    grouping on ``index`` and concatenating ``arguments``.
    """

    index: int
    id: str | None
    name: str | None
    arguments: str | None


@dataclass(slots=True, frozen=True)
class UsageEvent:
    usage: Usage


@dataclass(slots=True, frozen=True)
class Done:
    finish_reason: FinishReason | None


StreamEvent = TextDelta | ReasoningDelta | ToolCallDelta | UsageEvent | Done


def _events_for_chunk(chunk: ChatCompletionChunk) -> list[StreamEvent]:
    out: list[StreamEvent] = []
    choice = chunk.choices[0] if chunk.choices else None
    if choice is not None:
        delta = choice.delta
        if delta.content:
            out.append(TextDelta(content=delta.content))
        if delta.reasoning_content:
            out.append(ReasoningDelta(content=delta.reasoning_content))
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = getattr(tc, "index", None)
                out.append(
                    ToolCallDelta(
                        index=idx if isinstance(idx, int) else 0,
                        id=tc.id,
                        name=tc.function.name if tc.function else None,
                        arguments=tc.function.arguments if tc.function else None,
                    )
                )
    if chunk.usage is not None:
        out.append(UsageEvent(usage=chunk.usage))
    if choice is not None and choice.finish_reason is not None:
        try:
            fr = FinishReason(choice.finish_reason)
        except ValueError:
            fr = None
        out.append(Done(finish_reason=fr))
    return out


def stream_events(chunks: Iterator[ChatCompletionChunk]) -> Iterator[StreamEvent]:
    """Convert a sync chunk iterator to a typed event iterator."""
    for chunk in chunks:
        for ev in _events_for_chunk(chunk):
            yield ev


async def astream_events(
    chunks: AsyncIterator[ChatCompletionChunk],
) -> AsyncIterator[StreamEvent]:
    """Convert an async chunk iterator to a typed event iterator."""
    async for chunk in chunks:
        for ev in _events_for_chunk(chunk):
            yield ev


__all__ = [
    "Done",
    "ReasoningDelta",
    "StreamEvent",
    "TextDelta",
    "ToolCallDelta",
    "UsageEvent",
    "astream_events",
    "stream_events",
]
