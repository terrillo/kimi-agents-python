from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from ._enums import Model, Role
from ._stats import CacheStats, TokenStats
from .tools import (
    KimiTool,
    LoopGuards,
    _arun_tools_inner,
    _run_tools_inner,
)
from .types import (
    AssistantMessage,
    ChatCompletionChunk,
    ChoiceDelta,
    Message,
    ToolCall,
    Usage,
)

if TYPE_CHECKING:
    from .client import AsyncKimiClient, KimiClient


def _coerce_messages(records: Sequence[dict[str, Any] | Message]) -> list[Message]:
    out: list[Message] = []
    for r in records:
        out.append(r if isinstance(r, Message) else Message.model_validate(r))
    return out


def _copy_messages(messages: Sequence[Message]) -> list[Message]:
    return [m.model_copy(deep=True) for m in messages]


def _message_from_assistant(assistant: AssistantMessage) -> Message:
    """Convert an :class:`AssistantMessage` into a :class:`Message` for storage.

    Keeps ``reasoning_content`` so the next outbound turn includes it
    automatically — without this echo, thinking models return HTTP 400.
    """
    return Message(
        role=assistant.role,
        content=assistant.content,
        reasoning_content=assistant.reasoning_content,
        tool_calls=assistant.tool_calls,
    )


def _assistant_from_delta(
    content: str,
    reasoning_content: str,
    tool_calls: list[ToolCall] | None,
) -> Message:
    return Message(
        role=Role.ASSISTANT,
        content=content or None,
        reasoning_content=reasoning_content or None,
        tool_calls=tool_calls,
    )


class _BaseSession:
    """Shared state for :class:`Session` and :class:`AsyncSession`."""

    def __init__(
        self,
        client: Any,
        *,
        model: Model | str,
        system: str | None = None,
        defaults: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._model: Model | str = model
        self._system: str | None = system
        self._defaults: dict[str, Any] = dict(defaults or {})
        self._messages: list[Message] = []
        if system is not None:
            self._messages.append(Message(role=Role.SYSTEM, content=system))
        self.cache_stats: CacheStats = CacheStats()
        self.usage: TokenStats = TokenStats()
        self._checkpoints: dict[str, list[Message]] = {}

    @property
    def history(self) -> list[Message]:
        """Defensive copy of the current message list."""
        return list(self._messages)

    def append(self, message: Message | dict[str, Any]) -> None:
        """Append a raw message to history, bypassing send/stream bookkeeping.
        Useful for seeding few-shot examples."""
        self._messages.append(
            message if isinstance(message, Message) else Message.model_validate(message)
        )

    def reset(self) -> None:
        """Clear history (keeping the system prompt) and zero per-session stats.
        Checkpoints are also dropped."""
        self._messages = []
        if self._system is not None:
            self._messages.append(Message(role=Role.SYSTEM, content=self._system))
        self.cache_stats.reset()
        self.usage.reset()
        self._checkpoints.clear()

    def checkpoint(self) -> str:
        """Snapshot the current message list and return an opaque id."""
        cid = uuid.uuid4().hex
        self._checkpoints[cid] = _copy_messages(self._messages)
        return cid

    def restore(self, checkpoint_id: str) -> None:
        """Restore the message list from a previous :meth:`checkpoint` id.
        Raises :class:`KeyError` for unknown ids."""
        self._messages = _copy_messages(self._checkpoints[checkpoint_id])

    def checkpoints(self) -> list[str]:
        """Ids of all live checkpoints, in insertion order."""
        return list(self._checkpoints)

    def _merge_kwargs(self, overrides: dict[str, Any]) -> dict[str, Any]:
        return {**self._defaults, **overrides}

    def _append_user(self, content: str | None) -> None:
        if content is not None:
            self._messages.append(Message(role=Role.USER, content=content))

    def _append_assistant(self, assistant: AssistantMessage) -> Message:
        msg = _message_from_assistant(assistant)
        self._messages.append(msg)
        return msg

    def _record_usage(self, usage: Usage | None) -> None:
        self.cache_stats.record(usage, model=self._model)
        self.usage.record(usage, model=self._model)


class Session(_BaseSession):
    """A stateful, sync conversation that owns its message list, auto-echoes
    ``reasoning_content`` between turns, and tracks per-session cache and
    token usage.

    :meth:`send` handles plain chat or, when ``tools=[...]`` is given, drives
    the tool-call loop via the existing :func:`run_tools` machinery.
    :meth:`stream` yields raw :class:`ChatCompletionChunk` events and
    reassembles the assistant message into history when the stream completes.
    """

    def __init__(
        self,
        client: KimiClient,
        *,
        model: Model | str,
        system: str | None = None,
        **defaults: Any,
    ) -> None:
        super().__init__(client, model=model, system=system, defaults=defaults)

    def send(
        self,
        content: str | None = None,
        *,
        tools: Sequence[KimiTool] | None = None,
        max_steps: int = 5,
        guards: LoopGuards | None = None,
        **overrides: Any,
    ) -> Message:
        """Send a user turn and return the final assistant :class:`Message`.

        When ``tools`` is provided the call drives the assistant → tool →
        assistant loop to completion; the full transcript replaces
        ``self._messages`` so history stays consistent.
        """
        self._append_user(content)
        kwargs = self._merge_kwargs(overrides)
        if tools:
            response, transcript, _, _ = _run_tools_inner(
                self._client,
                model=self._model,
                messages=self._messages,
                tools=tools,
                max_steps=max_steps,
                guards=guards,
                **kwargs,
            )
            self._messages = _coerce_messages(transcript)
            assert response is not None
            self._record_usage(response.usage)
            return self._messages[-1]
        response = self._client.chat._create(
            model=self._model, messages=self._messages, **kwargs
        )
        self._record_usage(response.usage)
        return self._append_assistant(response.choices[0].message)

    def stream(
        self,
        content: str | None = None,
        **overrides: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Stream a user turn, yielding raw chunks. After the iterator is
        exhausted the reassembled assistant message is appended to history
        and per-session stats are updated from the terminal chunk's usage."""
        self._append_user(content)
        kwargs = self._merge_kwargs(overrides)
        return self._stream_impl(kwargs)

    def _stream_impl(self, kwargs: dict[str, Any]) -> Iterator[ChatCompletionChunk]:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        last_tool_calls: list[ToolCall] | None = None
        terminal_usage: Usage | None = None
        for chunk in self._client.chat._create(
            model=self._model, messages=self._messages, stream=True, **kwargs
        ):
            delta = chunk.choices[0].delta if chunk.choices else ChoiceDelta()
            if delta.content:
                content_parts.append(delta.content)
            if delta.reasoning_content:
                reasoning_parts.append(delta.reasoning_content)
            if delta.tool_calls:
                last_tool_calls = delta.tool_calls
            if chunk.usage is not None:
                terminal_usage = chunk.usage
            yield chunk
        self._messages.append(
            _assistant_from_delta(
                "".join(content_parts),
                "".join(reasoning_parts),
                last_tool_calls,
            )
        )
        self._record_usage(terminal_usage)

    def fork(self) -> Session:
        """Return a new :class:`Session` sharing the same client and defaults,
        with deep-copied messages and fresh stats / checkpoint maps. Use for
        tree-of-thought branching where each branch needs independent history."""
        child = Session(
            self._client, model=self._model, system=self._system, **self._defaults
        )
        child._messages = _copy_messages(self._messages)
        return child

    def estimated_tokens(self, content: str | None = None) -> int:
        """Pre-flight token count for ``history (+ optional pending user turn)``.

        Calls ``client.tokenizers.estimate`` against the current message list,
        optionally appending a draft user turn. Useful for guarding against
        ``MODEL_SPECS[model].context_length`` before issuing a request.
        """
        msgs = list(self._messages)
        if content is not None:
            msgs.append(Message(role=Role.USER, content=content))
        return self._client.tokenizers.estimate(
            model=self._model, messages=msgs
        ).data.total_tokens

    def stream_events(
        self,
        content: str | None = None,
        **overrides: Any,
    ) -> Iterator[Any]:
        """Stream a user turn as typed events (TextDelta, ReasoningDelta, …).

        Mirrors :meth:`stream` but yields :class:`~kimi_agents_python.events.StreamEvent`
        objects. History and per-session stats are still updated from the
        underlying chunks when the iterator is exhausted.
        """
        from .events import stream_events

        chunks = self.stream(content, **overrides)
        yield from stream_events(chunks)


class AsyncSession(_BaseSession):
    """Async counterpart to :class:`Session`."""

    def __init__(
        self,
        client: AsyncKimiClient,
        *,
        model: Model | str,
        system: str | None = None,
        **defaults: Any,
    ) -> None:
        super().__init__(client, model=model, system=system, defaults=defaults)

    async def send(
        self,
        content: str | None = None,
        *,
        tools: Sequence[KimiTool] | None = None,
        max_steps: int = 5,
        guards: LoopGuards | None = None,
        **overrides: Any,
    ) -> Message:
        self._append_user(content)
        kwargs = self._merge_kwargs(overrides)
        if tools:
            response, transcript, _, _ = await _arun_tools_inner(
                self._client,
                model=self._model,
                messages=self._messages,
                tools=tools,
                max_steps=max_steps,
                guards=guards,
                **kwargs,
            )
            self._messages = _coerce_messages(transcript)
            assert response is not None
            self._record_usage(response.usage)
            return self._messages[-1]
        response = await self._client.chat._create(
            model=self._model, messages=self._messages, **kwargs
        )
        self._record_usage(response.usage)
        return self._append_assistant(response.choices[0].message)

    def stream(
        self,
        content: str | None = None,
        **overrides: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        self._append_user(content)
        kwargs = self._merge_kwargs(overrides)
        return self._stream_impl(kwargs)

    async def _stream_impl(
        self, kwargs: dict[str, Any]
    ) -> AsyncIterator[ChatCompletionChunk]:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        last_tool_calls: list[ToolCall] | None = None
        terminal_usage: Usage | None = None
        stream = await self._client.chat._create(
            model=self._model, messages=self._messages, stream=True, **kwargs
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else ChoiceDelta()
            if delta.content:
                content_parts.append(delta.content)
            if delta.reasoning_content:
                reasoning_parts.append(delta.reasoning_content)
            if delta.tool_calls:
                last_tool_calls = delta.tool_calls
            if chunk.usage is not None:
                terminal_usage = chunk.usage
            yield chunk
        self._messages.append(
            _assistant_from_delta(
                "".join(content_parts),
                "".join(reasoning_parts),
                last_tool_calls,
            )
        )
        self._record_usage(terminal_usage)

    def fork(self) -> AsyncSession:
        child = AsyncSession(
            self._client, model=self._model, system=self._system, **self._defaults
        )
        child._messages = _copy_messages(self._messages)
        return child

    async def estimated_tokens(self, content: str | None = None) -> int:
        """Async counterpart to :meth:`Session.estimated_tokens`."""
        msgs = list(self._messages)
        if content is not None:
            msgs.append(Message(role=Role.USER, content=content))
        est = await self._client.tokenizers.estimate(
            model=self._model, messages=msgs
        )
        return est.data.total_tokens

    async def stream_events(
        self,
        content: str | None = None,
        **overrides: Any,
    ) -> AsyncIterator[Any]:
        """Async counterpart to :meth:`Session.stream_events`."""
        from .events import astream_events

        chunks = self.stream(content, **overrides)
        async for ev in astream_events(chunks):
            yield ev
