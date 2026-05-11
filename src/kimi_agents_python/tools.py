from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, get_type_hints, overload

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from .exceptions import KimiToolLoopError
from .types import FunctionDef, ToolDef

if TYPE_CHECKING:
    from ._enums import Model
    from .client import AsyncKimiClient, KimiClient
    from .types import ChatCompletion, Message


_SENTINEL = object()


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


def run_tools(
    client: KimiClient,
    *,
    model: Model | str,
    messages: Sequence[dict | Message],
    tools: Sequence[KimiTool],
    max_steps: int = 5,
    **chat_kwargs: Any,
) -> ChatCompletion:
    """Drive the chat → tool_calls → result loop synchronously to completion.

    Stops when the assistant returns a message with no ``tool_calls``.
    Raises :class:`KimiToolLoopError` if ``max_steps`` is exhausted first.
    ``can_parallel`` is recorded on each tool but has no effect in the sync
    helper — call dispatch is always sequential.
    """
    registry = {t.name: t for t in tools}
    convo = _serialise_messages(messages)
    for _ in range(max_steps):
        response = client.chat.create(
            model=model, messages=convo, tools=list(tools), **chat_kwargs
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            return response
        convo.append(_assistant_record(msg))
        for tc in msg.tool_calls:
            tool = registry.get(tc.function.name)
            content = (
                tool.invoke(tc.function.arguments)
                if tool is not None
                else f"Unknown tool: {tc.function.name}"
            )
            convo.append(_tool_result_record(tc, content))
    raise KimiToolLoopError(f"Tool loop exceeded max_steps={max_steps}")


async def arun_tools(
    client: AsyncKimiClient,
    *,
    model: Model | str,
    messages: Sequence[dict | Message],
    tools: Sequence[KimiTool],
    max_steps: int = 5,
    **chat_kwargs: Any,
) -> ChatCompletion:
    """Async equivalent of :func:`run_tools` with partitioned parallel dispatch.

    Each turn's ``tool_calls`` are split into two buckets: every call whose
    resolved tool has ``can_parallel=True`` is dispatched concurrently in a
    single :func:`asyncio.gather`, then any remaining calls (``can_parallel=False``
    tools and unknown tool names) run sequentially. Results are appended to the
    transcript in the model's original ``tool_calls`` order regardless of
    dispatch order, so subsequent turns see a deterministic conversation.
    """
    registry = {t.name: t for t in tools}
    convo = _serialise_messages(messages)
    for _ in range(max_steps):
        response = await client.chat.create(
            model=model, messages=convo, tools=list(tools), **chat_kwargs
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            return response
        convo.append(_assistant_record(msg))

        resolved = [(tc, registry.get(tc.function.name)) for tc in msg.tool_calls]
        results: list[str | None] = [None] * len(resolved)

        parallel_indices: list[int] = []
        parallel_coros: list[Awaitable[str]] = []
        serial_indices: list[int] = []
        for idx, (tc, t) in enumerate(resolved):
            if t is not None and t.can_parallel:
                parallel_indices.append(idx)
                parallel_coros.append(t.ainvoke(tc.function.arguments))
            else:
                serial_indices.append(idx)

        if parallel_coros:
            batch = await asyncio.gather(*parallel_coros)
            for idx, content in zip(parallel_indices, batch):
                results[idx] = content

        for idx in serial_indices:
            tc, t = resolved[idx]
            if t is None:
                results[idx] = f"Unknown tool: {tc.function.name}"
            else:
                results[idx] = await t.ainvoke(tc.function.arguments)

        for (tc, _), content in zip(resolved, results):
            convo.append(_tool_result_record(tc, content or ""))
    raise KimiToolLoopError(f"Tool loop exceeded max_steps={max_steps}")
