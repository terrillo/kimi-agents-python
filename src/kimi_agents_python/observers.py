"""Lifecycle hooks for the tool loop.

The engine's tool loop (``run_tools`` / ``arun_tools`` / ``Runner.run``) is
otherwise opaque between turns: callers that want to record what each turn
sent and got back, surface tool-call durations to a UI, or stream status to
a client end up wrapping every tool just to add instrumentation.

A :class:`RunObserver` replaces that wrapping. Subclass and override the
hooks you care about; the engine calls each one inline at the appropriate
point in the loop. Every method has a no-op default — the loop resolves
which hooks are actually overridden up front, so unset hooks pay no
per-turn cost.

Async-friendly: any hook may be a coroutine (``async def``) when the agent
runs through the async loop. Passing async hooks to the sync loop is an
error and raises :class:`TypeError` rather than silently dropping events.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._stats import TokenStats
    from .types import ChatCompletion, Usage


class RunObserver:
    """Default no-op observer; subclass and override the hooks you want.

    Hook ordering for one turn of :func:`arun_tools` / :func:`run_tools`:

    1. :meth:`on_turn_start` — fired with ``step`` and the message list that
       is *about to be sent* (post-compaction).
    2. :meth:`on_compaction` — fired only if a ``compactor`` rewrote the
       payload, with the wire-form message counts before and after.
    3. The chat call happens.
    4. :meth:`on_chat_response` — fired with the raw :class:`ChatCompletion`
       and the cumulative :class:`TokenStats` after this turn.
    5. For each tool call the model emitted, :meth:`on_tool_call_start`
       then :meth:`on_tool_call_finished`. In the async loop these are
       fired *inside* each tool task, so the pair around a slow tool does
       not block the pair around a fast sibling.
    6. :meth:`on_step_complete` — fired once per turn after every tool
       result has been appended.

    When the loop exits cleanly the terminal turn (no tool calls) still
    fires steps 1-4 and then :meth:`on_step_complete`. There is no
    ``on_run_complete`` hook here — :class:`Runner` owns that signal
    because only it knows about ``truncated`` and the final ``RunResult``.
    """

    def on_turn_start(
        self, *, step: int, messages: list[dict[str, Any]]
    ) -> Any: ...

    def on_compaction(
        self, *, step: int, before: int, after: int
    ) -> Any: ...

    def on_chat_response(
        self,
        *,
        step: int,
        response: "ChatCompletion",
        usage: "Usage | None",
        usage_so_far: "TokenStats",
    ) -> Any: ...

    def on_tool_call_start(
        self,
        *,
        step: int,
        call_id: str,
        tool_name: str,
        arguments: str,
    ) -> Any: ...

    def on_tool_call_finished(
        self,
        *,
        step: int,
        call_id: str,
        tool_name: str,
        arguments: str,
        result: str,
        duration_ms: int,
    ) -> Any: ...

    def on_step_complete(
        self, *, step: int, usage_so_far: "TokenStats"
    ) -> Any: ...


_HOOK_NAMES: tuple[str, ...] = (
    "on_turn_start",
    "on_compaction",
    "on_chat_response",
    "on_tool_call_start",
    "on_tool_call_finished",
    "on_step_complete",
)


@dataclass(slots=True)
class _HookSet:
    """Pre-resolved hooks. Each attribute is the bound override or ``None``.

    Resolved once per run by :func:`_resolve_hooks` so the loop's per-turn
    check is a single ``is not None`` per hook instead of a bound-method
    allocation + no-op dispatch.
    """

    on_turn_start: Callable[..., Any] | None
    on_compaction: Callable[..., Any] | None
    on_chat_response: Callable[..., Any] | None
    on_tool_call_start: Callable[..., Any] | None
    on_tool_call_finished: Callable[..., Any] | None
    on_step_complete: Callable[..., Any] | None


_EMPTY_HOOKS = _HookSet(None, None, None, None, None, None)


def _resolve_hooks(observer: RunObserver | None) -> _HookSet:
    """Return a :class:`_HookSet` with bound methods for every overridden hook.

    A hook counts as overridden when the observer's class defines a
    different function for it than :class:`RunObserver` does — so a
    no-argument ``RunObserver()`` resolves to all-``None`` and pays no
    per-turn cost.
    """
    if observer is None:
        return _EMPTY_HOOKS
    cls = type(observer)
    base = RunObserver
    resolved: dict[str, Callable[..., Any] | None] = {}
    for name in _HOOK_NAMES:
        if getattr(cls, name, None) is not getattr(base, name):
            resolved[name] = getattr(observer, name)
        else:
            resolved[name] = None
    return _HookSet(**resolved)


def _call_sync(method: Callable[..., Any], /, **kwargs: Any) -> None:
    """Invoke a sync observer hook.

    Raises :class:`TypeError` if the hook returns a coroutine — async
    observers belong on the async loop, and silent drop would hide a real
    instrumentation bug.
    """
    result = method(**kwargs)
    if asyncio.iscoroutine(result):
        # Close to suppress the "never awaited" warning before raising.
        result.close()
        raise TypeError(
            f"Observer hook {method.__qualname__} is async; the sync tool "
            "loop cannot await it. Use Runner.run / arun_tools instead."
        )


async def _call_async(method: Callable[..., Any], /, **kwargs: Any) -> None:
    """Invoke an async-or-sync observer hook from the async loop."""
    result = method(**kwargs)
    if asyncio.iscoroutine(result):
        await result


__all__ = ["RunObserver"]
