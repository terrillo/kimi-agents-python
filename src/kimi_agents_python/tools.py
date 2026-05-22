from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, get_type_hints, overload, runtime_checkable

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from ._stats import TokenStats
from .exceptions import (
    KimiToolLoopError,
    ReadOnlyStreakExceededError,
    RepeatedToolCallError,
    TokenBudgetExceededError,
)
from .observers import RunObserver, _HookSet, _call_async, _call_sync, _resolve_hooks
from .types import FunctionDef, ToolDef

if TYPE_CHECKING:
    from ._enums import Model
    from .client import AsyncKimiClient, KimiClient
    from .types import ChatCompletion, Message, Usage


_SENTINEL = object()


@runtime_checkable
class ToolLike(Protocol):
    """Anything the chat / tool-loop dispatcher accepts as a tool.

    Implemented by :class:`KimiTool` (user functions), :class:`BuiltinTool`
    (server-side builtins like ``$web_search``), and :class:`FormulaTool`
    (official Moonshot Formula API tools).
    """

    name: str
    read_only: bool
    can_parallel: bool

    def to_tool_def(self) -> Any: ...
    def invoke(self, arguments: str | dict[str, Any]) -> str: ...
    async def ainvoke(self, arguments: str | dict[str, Any]) -> str: ...


def _default_failure(tool_name: str) -> Callable[[Exception], str]:
    def fmt(e: Exception) -> str:
        return f"Tool '{tool_name}' raised {type(e).__name__}: {e}"

    return fmt


def _strip_titles(schema: dict[str, Any]) -> dict[str, Any]:
    schema.pop("title", None)
    for prop in (schema.get("properties") or {}).values():
        if isinstance(prop, dict):
            prop.pop("title", None)
    return schema


def _split_annotated(annotation: Any) -> tuple[Any, FieldInfo | None]:
    """If ``annotation`` is ``Annotated[T, ..., Field(...)]`` return ``(T, FieldInfo)``."""
    if hasattr(annotation, "__metadata__"):
        type_ = annotation.__origin__
        for meta in annotation.__metadata__:
            if isinstance(meta, FieldInfo):
                return type_, meta
        return type_, None
    return annotation, None


def _build_schema(func: Callable[..., Any]) -> tuple[dict[str, Any], type[BaseModel]]:
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception as e:
        raise TypeError(
            f"@kimi_tool({func.__name__}): cannot resolve type hints: {e}"
        ) from e

    fields: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"@kimi_tool({func.__name__}): *args / **kwargs parameters "
                f"are not supported."
            )
        if name not in hints:
            raise TypeError(
                f"@kimi_tool({func.__name__}): parameter '{name}' is missing "
                f"a type annotation."
            )
        type_, field_info = _split_annotated(hints[name])
        has_default = param.default is not inspect.Parameter.empty
        if field_info is None:
            fields[name] = (type_, param.default if has_default else ...)
        else:
            if has_default:
                field_info = FieldInfo.merge_field_infos(
                    field_info, FieldInfo(default=param.default)
                )
            fields[name] = (type_, field_info)

    model = create_model(f"{func.__name__}_Args", **fields)
    return _strip_titles(model.model_json_schema()), model


@dataclass(slots=True)
class KimiTool:
    """A Python function wrapped as a Kimi function-calling tool.

    Build via the :func:`kimi_tool` decorator. Pass instances directly to
    ``client.chat(tools=[...])`` or to :func:`run_tools` / :func:`arun_tools`.
    """

    name: str
    description: str
    params_json_schema: dict[str, Any]
    func: Callable[..., Any]
    is_async: bool
    strict: bool
    can_parallel: bool
    """Client-side metadata; never serialised onto the Kimi request body.
    Read by :func:`arun_tools` to decide which calls in a turn go into
    ``asyncio.gather`` and which run sequentially."""
    read_only: bool
    """Marker for :class:`LoopGuards` — when ``True``, this tool does not
    mutate state, and a streak of consecutive read-only calls counts toward
    ``read_only_streak``. Never serialised on the wire."""
    failure_error_function: Callable[[Exception], str] | None
    args_model: type[BaseModel] = field(repr=False)

    def to_tool_def(self) -> ToolDef:
        """Build the wire-format ToolDef. ``can_parallel`` is intentionally omitted."""
        return ToolDef(
            function=FunctionDef(
                name=self.name,
                description=self.description or None,
                parameters=self.params_json_schema,
                strict=self.strict or None,
            )
        )

    def invoke(self, arguments: str | dict[str, Any]) -> str:
        if self.is_async:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.ainvoke(arguments))
            raise RuntimeError(
                f"Tool '{self.name}' is async; call `await tool.ainvoke(...)` "
                f"from an async context, or use `arun_tools(...)`."
            )
        try:
            kwargs = self._validate(arguments)
            return self._format_result(self.func(**kwargs))
        except Exception as e:
            return self._handle_failure(e)

    async def ainvoke(self, arguments: str | dict[str, Any]) -> str:
        try:
            kwargs = self._validate(arguments)
            result = self.func(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            return self._format_result(result)
        except Exception as e:
            return self._handle_failure(e)

    def _validate(self, arguments: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(arguments, str):
            raw = arguments.strip()
            data: Any = json.loads(raw) if raw else {}
        else:
            data = arguments
        return self.args_model.model_validate(data).model_dump()

    def _format_result(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            return str(result)

    def _handle_failure(self, exc: Exception) -> str:
        if self.failure_error_function is None:
            raise exc
        return self.failure_error_function(exc)


@overload
def kimi_tool(func: Callable[..., Any], /) -> KimiTool: ...
@overload
def kimi_tool(
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    strict: bool = True,
    can_parallel: bool = True,
    read_only: bool = False,
    failure_error_function: Callable[[Exception], str] | None = ...,
    use_docstring: bool = True,
) -> Callable[[Callable[..., Any]], KimiTool]: ...
def kimi_tool(
    func: Callable[..., Any] | None = None,
    /,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    strict: bool = True,
    can_parallel: bool = True,
    read_only: bool = False,
    failure_error_function: Callable[[Exception], str] | None | object = _SENTINEL,
    use_docstring: bool = True,
) -> KimiTool | Callable[[Callable[..., Any]], KimiTool]:
    """Turn a Python function into a Kimi tool.

    Works bare (``@kimi_tool``) or parameterised (``@kimi_tool(strict=False)``).
    The function's signature drives the JSON schema; per-parameter descriptions
    come from ``Annotated[T, Field(description=...)]``. The first line of the
    docstring becomes the tool description unless ``description_override`` is set.
    """

    def wrap(f: Callable[..., Any]) -> KimiTool:
        schema, args_model = _build_schema(f)
        name = name_override or f.__name__
        if description_override is not None:
            description = description_override
        elif use_docstring and f.__doc__:
            description = f.__doc__.strip().split("\n", 1)[0].strip()
        else:
            description = ""
        failure: Callable[[Exception], str] | None
        if failure_error_function is _SENTINEL:
            failure = _default_failure(name)
        else:
            failure = failure_error_function  # type: ignore[assignment]
        return KimiTool(
            name=name,
            description=description,
            params_json_schema=schema,
            func=f,
            is_async=inspect.iscoroutinefunction(f),
            strict=strict,
            can_parallel=can_parallel,
            read_only=read_only,
            failure_error_function=failure,
            args_model=args_model,
        )

    if func is not None:
        return wrap(func)
    return wrap


# --- Auto tool-call loops ------------------------------------------------------


def _serialise_messages(messages: Sequence[dict | Any]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if isinstance(m, dict):
            out.append(m)
        else:
            out.append(m.model_dump(exclude_none=True, by_alias=True, mode="json"))
    return out


def _assistant_record(msg: Any) -> dict:
    return msg.model_dump(exclude_none=True, by_alias=True, mode="json")


def _tool_result_record(tc: Any, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tc.id,
        "name": tc.function.name,
        "content": content,
    }


Compactor = Callable[[list[dict]], list[dict]]
"""Optional per-turn transcript compactor for the tool loop.

Called with the full running conversation (list of API-shaped message
dicts) immediately before each ``chat._create`` and must return the message
list to actually send. The loop keeps accumulating the *full* transcript
internally (it is what the returned transcript / ``RunResult.messages``
reflect); only the per-call payload is replaced by the compactor's output.
The callable MUST be pure (not mutate its argument) and MUST return an
API-valid message list (system first; every ``tool`` message still preceded
by the assistant message carrying its ``tool_call_id``). ``None`` disables
compaction — the full transcript is sent every turn, the prior behavior.
"""


@dataclass(frozen=True, slots=True)
class LoopGuards:
    """Optional safety limits for :func:`run_tools` / :func:`arun_tools`.

    Each field is opt-in: leaving it ``None`` disables that guard. Guards
    fire as :class:`KimiToolLoopError` subclasses so callers can catch the
    base class or the specific reason.
    """

    max_tokens: int | None = None
    """Raise :class:`TokenBudgetExceededError` once cumulative
    ``usage.total_tokens`` across all turns crosses this value."""

    read_only_streak: int | None = None
    """Raise :class:`ReadOnlyStreakExceededError` after this many consecutive
    calls to tools marked ``@kimi_tool(read_only=True)`` with no mutating call
    in between. Resets when a non-read-only tool runs."""

    repeat_threshold: int | None = None
    """Raise :class:`RepeatedToolCallError` when the same ``(tool_name,
    arguments)`` appears this many times in a row across the loop. Resets on
    any different call."""


def _normalize_args(arguments: str) -> str:
    """Best-effort canonicalize tool-call args for repeat detection.

    Returns a key-sorted JSON dump so cosmetically-different whitespace doesn't
    defeat the guard. Falls through to the raw value on malformed input.
    """
    try:
        return json.dumps(json.loads(arguments), sort_keys=True)
    except (json.JSONDecodeError, TypeError):
        return arguments


class _LoopState:
    """Per-loop counters that drive :class:`LoopGuards` checks."""

    __slots__ = ("guards", "total_tokens", "read_only_streak", "_recent")

    def __init__(self, guards: LoopGuards) -> None:
        self.guards = guards
        self.total_tokens = 0
        self.read_only_streak = 0
        self._recent: deque[tuple[str, str]] = deque(
            maxlen=guards.repeat_threshold or 1
        )

    def check_tokens(self, usage: Usage | None) -> KimiToolLoopError | None:
        """Update the cumulative token counter; return the cap exception or ``None``.

        Returning the exception (instead of raising it) lets the inner loop
        decide whether to ``raise`` it or break gracefully and attach the
        accumulated transcript and ``TokenStats`` to it — which is what the
        ``graceful_caps`` path does.
        """
        if self.guards.max_tokens is None or usage is None:
            return None
        self.total_tokens += usage.total_tokens
        if self.total_tokens > self.guards.max_tokens:
            return TokenBudgetExceededError(
                f"Tool loop exceeded token budget: "
                f"{self.total_tokens} > {self.guards.max_tokens}"
            )
        return None

    def check_call(
        self, tool: ToolLike | None, tc: Any
    ) -> KimiToolLoopError | None:
        """Record one tool-call attempt; return the cap exception or ``None``.

        Same return-don't-raise contract as :meth:`check_tokens` so the inner
        loop can choose between raising and graceful truncation.
        """
        threshold = self.guards.repeat_threshold
        if threshold is not None:
            key = (tc.function.name, _normalize_args(tc.function.arguments))
            self._recent.append(key)
            if len(self._recent) == threshold and len(set(self._recent)) == 1:
                return RepeatedToolCallError(
                    f"Tool '{tc.function.name}' called {threshold} times in a row "
                    f"with the same arguments"
                )
        streak_limit = self.guards.read_only_streak
        if streak_limit is None or tool is None:
            return None
        if tool.read_only:
            self.read_only_streak += 1
            if self.read_only_streak >= streak_limit:
                return ReadOnlyStreakExceededError(
                    f"Tool loop made {self.read_only_streak} consecutive read-only "
                    f"calls (limit {streak_limit}) without a mutating call"
                )
        else:
            self.read_only_streak = 0
        return None


def _trip(
    exc: KimiToolLoopError,
    *,
    graceful_caps: bool,
    convo: list[dict],
    usage_total: TokenStats,
) -> None:
    """Either raise ``exc`` with partial loop state attached, or return cleanly.

    Returning means the caller should break out of the loop and report
    ``truncated=True``. The transcript snapshot (``list(convo)``) is only
    paid when we actually raise — the graceful path discards ``exc``, so
    decorating it would be wasted allocation.
    """
    if graceful_caps:
        return
    exc.usage_so_far = usage_total
    exc.partial_transcript = list(convo)
    raise exc


def _run_tools_inner(
    client: KimiClient,
    *,
    model: Model | str,
    messages: Sequence[dict | Message],
    tools: Sequence[ToolLike],
    max_steps: int,
    guards: LoopGuards | None,
    compactor: Compactor | None = None,
    observer: RunObserver | None = None,
    graceful_caps: bool = False,
    **chat_kwargs: Any,
) -> tuple["ChatCompletion | None", list[dict], TokenStats, bool]:
    """Drive the sync tool loop and return the terminal response, the full
    assembled transcript (user + assistant + tool messages, all dict-shaped
    per the API payload form), cumulative token usage summed across every
    turn of the loop, and a ``truncated`` flag.

    The terminal assistant message is appended to the transcript before
    returning, so callers like :class:`~kimi_agents_python.session.Session`
    can persist the complete conversation. ``reasoning_content`` is preserved
    on every assistant turn via :func:`_assistant_record`. The returned
    transcript is always the full conversation; ``compactor``, when given,
    only rewrites the per-turn payload sent to the model (see
    :data:`Compactor`).

    When ``graceful_caps`` is ``True`` any :class:`LoopGuards` violation or
    ``max_steps`` exhaustion breaks out of the loop and returns the partial
    state with ``truncated=True`` instead of raising. The exception that
    would have been raised — carrying ``usage_so_far`` and
    ``partial_transcript`` — is discarded; callers wanting both partial
    state *and* the trip reason should use ``graceful_caps=False`` and catch
    :class:`KimiToolLoopError`.

    ``observer``, when given, receives :class:`RunObserver` lifecycle
    callbacks. Passing an observer with async hooks to this sync path
    raises :class:`TypeError` rather than silently dropping events.
    """
    registry = {t.name: t for t in tools}
    convo = _serialise_messages(messages)
    state = _LoopState(guards or LoopGuards())
    usage_total = TokenStats()
    hooks = _resolve_hooks(observer)
    tool_defs = list(tools)
    last_response: ChatCompletion | None = None
    for step in range(max_steps):
        payload = compactor(convo) if compactor is not None else convo
        if hooks.on_compaction is not None and compactor is not None and len(payload) != len(convo):
            _call_sync(hooks.on_compaction, step=step, before=len(convo), after=len(payload))
        if hooks.on_turn_start is not None:
            _call_sync(hooks.on_turn_start, step=step, messages=payload)
        response = client.chat._create(
            model=model, messages=payload, tools=tool_defs, **chat_kwargs
        )
        last_response = response
        # Record usage BEFORE the budget check so the exception (and the
        # on_chat_response hook) see the turn that pushed us over.
        usage_total.record(response.usage, model=model)
        if hooks.on_chat_response is not None:
            _call_sync(
                hooks.on_chat_response,
                step=step,
                response=response,
                usage=response.usage,
                usage_so_far=usage_total,
            )
        budget_exc = state.check_tokens(response.usage)
        msg = response.choices[0].message
        # Append the offending assistant turn before tripping the budget so
        # ``partial_transcript`` includes the turn that pushed us over.
        convo.append(_assistant_record(msg))
        if budget_exc is not None:
            _trip(budget_exc, graceful_caps=graceful_caps, convo=convo, usage_total=usage_total)
            return last_response, convo, usage_total, True
        if not msg.tool_calls:
            if hooks.on_step_complete is not None:
                _call_sync(hooks.on_step_complete, step=step, usage_so_far=usage_total)
            return response, convo, usage_total, False
        for tc in msg.tool_calls:
            tool = registry.get(tc.function.name)
            call_exc = state.check_call(tool, tc)
            if call_exc is not None:
                _trip(call_exc, graceful_caps=graceful_caps, convo=convo, usage_total=usage_total)
                return last_response, convo, usage_total, True
            args_str = tc.function.arguments
            if hooks.on_tool_call_start is not None:
                _call_sync(
                    hooks.on_tool_call_start,
                    step=step,
                    call_id=getattr(tc, "id", "") or "",
                    tool_name=tc.function.name,
                    arguments=args_str,
                )
            t0 = time.perf_counter()
            content = (
                tool.invoke(args_str)
                if tool is not None
                else f"Unknown tool: {tc.function.name}"
            )
            duration_ms = int((time.perf_counter() - t0) * 1000)
            convo.append(_tool_result_record(tc, content))
            if hooks.on_tool_call_finished is not None:
                _call_sync(
                    hooks.on_tool_call_finished,
                    step=step,
                    call_id=getattr(tc, "id", "") or "",
                    tool_name=tc.function.name,
                    arguments=args_str,
                    result=content,
                    duration_ms=duration_ms,
                )
        if hooks.on_step_complete is not None:
            _call_sync(hooks.on_step_complete, step=step, usage_so_far=usage_total)
    step_exc = KimiToolLoopError(f"Tool loop exceeded max_steps={max_steps}")
    _trip(step_exc, graceful_caps=graceful_caps, convo=convo, usage_total=usage_total)
    return last_response, convo, usage_total, True


def run_tools(
    client: KimiClient,
    *,
    model: Model | str,
    messages: Sequence[dict | Message],
    tools: Sequence[ToolLike],
    max_steps: int = 5,
    guards: LoopGuards | None = None,
    compactor: Compactor | None = None,
    observer: RunObserver | None = None,
    **chat_kwargs: Any,
) -> ChatCompletion:
    """Drive the chat → tool_calls → result loop synchronously to completion.

    Stops when the assistant returns a message with no ``tool_calls``.
    Raises :class:`KimiToolLoopError` if ``max_steps`` is exhausted first;
    raises a subclass when an optional :class:`LoopGuards` limit is hit. The
    raised exception carries ``usage_so_far`` and ``partial_transcript`` —
    no need to back-fill from a client-wide counter on the except branch.
    ``can_parallel`` is recorded on each tool but has no effect in the sync
    helper — call dispatch is always sequential. ``compactor`` optionally
    rewrites the per-turn payload sent to the model (see :data:`Compactor`).
    ``observer`` receives :class:`RunObserver` lifecycle callbacks.

    Graceful truncation lives on :class:`Runner.run` (via
    ``KimiAgent.graceful_caps``) because only ``RunResult`` carries the
    ``truncated`` flag; direct callers of this helper should catch
    :class:`KimiToolLoopError` and read ``exc.partial_transcript`` /
    ``exc.usage_so_far``.
    """
    response, _, _, _ = _run_tools_inner(
        client,
        model=model,
        messages=messages,
        tools=tools,
        max_steps=max_steps,
        guards=guards,
        compactor=compactor,
        observer=observer,
        **chat_kwargs,
    )
    assert response is not None
    return response


async def _arun_tools_inner(
    client: AsyncKimiClient,
    *,
    model: Model | str,
    messages: Sequence[dict | Message],
    tools: Sequence[ToolLike],
    max_steps: int,
    guards: LoopGuards | None,
    compactor: Compactor | None = None,
    observer: RunObserver | None = None,
    graceful_caps: bool = False,
    **chat_kwargs: Any,
) -> tuple["ChatCompletion | None", list[dict], TokenStats, bool]:
    """Async counterpart to :func:`_run_tools_inner`. Same contract: returns
    ``(terminal_response, full_transcript, cumulative_usage, truncated)``
    with the terminal assistant message included in the transcript and
    usage summed across every turn. ``compactor`` only rewrites the
    per-turn payload sent to the model. ``observer``, ``graceful_caps``,
    and the partial-state-on-exception behaviour mirror the sync inner
    (see :func:`_run_tools_inner`); async observer hooks are awaited.
    """
    registry = {t.name: t for t in tools}
    convo = _serialise_messages(messages)
    state = _LoopState(guards or LoopGuards())
    usage_total = TokenStats()
    hooks = _resolve_hooks(observer)
    tool_defs = list(tools)
    last_response: ChatCompletion | None = None
    for step in range(max_steps):
        payload = compactor(convo) if compactor is not None else convo
        if hooks.on_compaction is not None and compactor is not None and len(payload) != len(convo):
            await _call_async(hooks.on_compaction, step=step, before=len(convo), after=len(payload))
        if hooks.on_turn_start is not None:
            await _call_async(hooks.on_turn_start, step=step, messages=payload)
        response = await client.chat._create(
            model=model, messages=payload, tools=tool_defs, **chat_kwargs
        )
        last_response = response
        usage_total.record(response.usage, model=model)
        if hooks.on_chat_response is not None:
            await _call_async(
                hooks.on_chat_response,
                step=step,
                response=response,
                usage=response.usage,
                usage_so_far=usage_total,
            )
        budget_exc = state.check_tokens(response.usage)
        msg = response.choices[0].message
        convo.append(_assistant_record(msg))
        if budget_exc is not None:
            _trip(budget_exc, graceful_caps=graceful_caps, convo=convo, usage_total=usage_total)
            return last_response, convo, usage_total, True
        if not msg.tool_calls:
            if hooks.on_step_complete is not None:
                await _call_async(hooks.on_step_complete, step=step, usage_so_far=usage_total)
            return response, convo, usage_total, False

        resolved = [(tc, registry.get(tc.function.name)) for tc in msg.tool_calls]
        # Evaluate all per-call guards BEFORE dispatch so a trip aborts the
        # whole batch and no tool side effects run for the offending turn.
        for tc, t in resolved:
            call_exc = state.check_call(t, tc)
            if call_exc is not None:
                _trip(call_exc, graceful_caps=graceful_caps, convo=convo, usage_total=usage_total)
                return last_response, convo, usage_total, True
        results: list[str | None] = [None] * len(resolved)

        async def _invoke_with_hooks(idx: int, t: ToolLike | None, tc: Any) -> None:
            call_id = getattr(tc, "id", "") or ""
            args_str = tc.function.arguments
            if hooks.on_tool_call_start is not None:
                await _call_async(
                    hooks.on_tool_call_start,
                    step=step,
                    call_id=call_id,
                    tool_name=tc.function.name,
                    arguments=args_str,
                )
            t0 = time.perf_counter()
            if t is None:
                content = f"Unknown tool: {tc.function.name}"
            else:
                content = await t.ainvoke(args_str)
            duration_ms = int((time.perf_counter() - t0) * 1000)
            results[idx] = content
            if hooks.on_tool_call_finished is not None:
                await _call_async(
                    hooks.on_tool_call_finished,
                    step=step,
                    call_id=call_id,
                    tool_name=tc.function.name,
                    arguments=args_str,
                    result=content,
                    duration_ms=duration_ms,
                )

        parallel_indices: list[int] = []
        serial_indices: list[int] = []
        for idx, (tc, t) in enumerate(resolved):
            if t is not None and t.can_parallel:
                parallel_indices.append(idx)
            else:
                serial_indices.append(idx)

        # KimiTool.ainvoke rewrites tool exceptions via failure_error_function;
        # the TaskGroup only ever fires for tools that opted out, which is the
        # case we want to bail on.
        if parallel_indices:
            async with asyncio.TaskGroup() as tg:
                for idx in parallel_indices:
                    tc, t = resolved[idx]
                    tg.create_task(
                        _invoke_with_hooks(idx, t, tc),
                        name=f"tool:{tc.function.name}",
                    )

        for idx in serial_indices:
            tc, t = resolved[idx]
            await _invoke_with_hooks(idx, t, tc)

        for (tc, _), content in zip(resolved, results):
            convo.append(_tool_result_record(tc, content or ""))
        if hooks.on_step_complete is not None:
            await _call_async(hooks.on_step_complete, step=step, usage_so_far=usage_total)
    step_exc = KimiToolLoopError(f"Tool loop exceeded max_steps={max_steps}")
    _trip(step_exc, graceful_caps=graceful_caps, convo=convo, usage_total=usage_total)
    return last_response, convo, usage_total, True


async def arun_tools(
    client: AsyncKimiClient,
    *,
    model: Model | str,
    messages: Sequence[dict | Message],
    tools: Sequence[ToolLike],
    max_steps: int = 5,
    guards: LoopGuards | None = None,
    compactor: Compactor | None = None,
    observer: RunObserver | None = None,
    **chat_kwargs: Any,
) -> ChatCompletion:
    """Async equivalent of :func:`run_tools` with partitioned parallel dispatch.

    Each turn's ``tool_calls`` are split into two buckets: every call whose
    resolved tool has ``can_parallel=True`` is dispatched concurrently in a
    single :class:`asyncio.TaskGroup`, then any remaining calls
    (``can_parallel=False`` tools and unknown tool names) run sequentially.
    Results are appended to the transcript in the model's original
    ``tool_calls`` order regardless of dispatch order, so subsequent turns
    see a deterministic conversation.

    :class:`LoopGuards` checks are evaluated in the model's original call
    order, before any dispatch — so a guard can short-circuit a parallel
    batch before it kicks off.

    ``observer`` mirrors :func:`run_tools`; async observer methods are
    awaited inline. Graceful truncation lives on :class:`Runner.run`
    only — see the note on :func:`run_tools`.
    """
    response, _, _, _ = await _arun_tools_inner(
        client,
        model=model,
        messages=messages,
        tools=tools,
        max_steps=max_steps,
        guards=guards,
        compactor=compactor,
        observer=observer,
        **chat_kwargs,
    )
    assert response is not None
    return response
