from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterable, Iterator
from contextlib import AsyncExitStack, ExitStack
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import httpx
from pydantic import BaseModel, TypeAdapter

from .._enums import FinishReason, Model, Role
from .._http import parse_sse_line, raise_for_status
from .._retry import retry_async, retry_sync
from ..exceptions import ManualMultiTurnError, StructuredParseError
from ..specs import get_model_spec
from ..types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Message,
    ParsedChatCompletion,
    PrefilledCompletion,
)

if TYPE_CHECKING:
    from ..client import AsyncKimiClient, KimiClient


_MessageInput = Message | dict[str, Any]


_SESSION_HINT = (
    "Use Session for multi-turn flows so reasoning_content and tool "
    "transcripts are echoed correctly: `from kimi_agents_python import Session`."
)


def _message_role(m: _MessageInput) -> Role | None:
    raw = m.role if isinstance(m, Message) else m.get("role")
    if raw is None:
        return None
    try:
        return Role(raw)
    except ValueError:
        return None


def _is_partial_prefill(m: _MessageInput) -> bool:
    partial = m.partial if isinstance(m, Message) else m.get("partial")
    return partial is True


def _enforce_single_turn(messages: list[_MessageInput]) -> None:
    """Reject manual multi-turn payloads — Session owns multi-turn state.

    Allowed shapes for raw ``chat.create()``: ``[system?, user]`` or, for
    prefill mode, ``[system?, user, assistant(partial=True)]``.
    """
    last = len(messages) - 1
    for i, m in enumerate(messages):
        role = _message_role(m)
        if role is Role.TOOL:
            raise ManualMultiTurnError(
                f"chat.create() refuses payloads containing tool results. "
                f"{_SESSION_HINT}"
            )
        if role is Role.ASSISTANT and not (i == last and _is_partial_prefill(m)):
            raise ManualMultiTurnError(
                f"chat.create() refuses payloads with a prior assistant "
                f"message. {_SESSION_HINT}"
            )


def _allowed_chat_params() -> frozenset[str]:
    keys: set[str] = set()
    for name, field in ChatCompletionRequest.model_fields.items():
        keys.add(name)
        if field.alias:
            keys.add(field.alias)
    return frozenset(keys - {"model", "messages", "stream"})


_ALLOWED_CHAT_PARAMS = _allowed_chat_params()

T = TypeVar("T")


def _resolve_schema(
    response_format: type[BaseModel] | TypeAdapter[Any] | Any,
) -> tuple[dict[str, Any], str, Any]:
    """Return (json_schema, name, parser) for chat.parse().

    Accepts a Pydantic ``BaseModel`` subclass, a ``TypeAdapter``, or any type
    that ``TypeAdapter`` can wrap (e.g. ``list[int]``, ``dict[str, Foo]``).
    """
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        cls = response_format
        return (
            cls.model_json_schema(),
            cls.__name__,
            cls.model_validate_json,
        )
    adapter: TypeAdapter[Any]
    name = "Response"
    if isinstance(response_format, TypeAdapter):
        adapter = response_format
    else:
        adapter = TypeAdapter(response_format)
        rt = getattr(response_format, "__name__", None)
        if isinstance(rt, str):
            name = rt
    return (
        adapter.json_schema(),
        name,
        lambda s: adapter.validate_python(json.loads(s)),
    )


def _parse_completion(
    completion: ChatCompletion, parser: Any
) -> ParsedChatCompletion[Any]:
    """Run ``parser`` over the completion's text, with actionable errors.

    Replaces the opaque ``Invalid JSON: EOF`` pydantic raises when the model
    output was truncated or empty with a message that says what to change.
    """
    choice = completion.choices[0]
    content = choice.message.content
    if choice.finish_reason == FinishReason.LENGTH:
        raise StructuredParseError(
            "chat.parse(): response hit max_tokens before the JSON was "
            "complete, so it can't be parsed. Raise max_tokens and retry.",
            completion=completion,
        )
    if not content or not content.strip():
        raise StructuredParseError(
            "chat.parse(): the model returned empty content. For "
            "thinking-capable models the token budget may have been spent on "
            "reasoning — disable thinking for this call or raise max_tokens.",
            completion=completion,
        )
    try:
        parsed = parser(content)
    except Exception as e:  # noqa: BLE001 - re-raised with context below
        raise StructuredParseError(
            f"chat.parse(): model output is not valid JSON for the requested "
            f"schema: {e}",
            completion=completion,
        ) from e
    return ParsedChatCompletion(completion=completion, parsed=parsed)


def _coerce_tool(t: Any) -> dict[str, Any]:
    """Convert a tool entry to the wire-format dict.

    Accepts: a pre-built dict, a Pydantic ToolDef/BuiltinToolDef, or any object
    exposing ``to_tool_def()``. Pydantic models are dumped with aliases applied
    so ``parameters`` etc. land at the right keys.
    """
    if hasattr(t, "to_tool_def"):
        t = t.to_tool_def()
    if hasattr(t, "model_dump"):
        return t.model_dump(exclude_none=True, by_alias=True, mode="json")
    return t


def _build_request_body(
    *,
    model: Model | str,
    messages: Iterable[_MessageInput],
    stream: bool,
    extra: dict[str, Any],
    default_cache_key: str | None = None,
) -> dict[str, Any]:
    unknown = set(extra) - _ALLOWED_CHAT_PARAMS
    if unknown:
        raise TypeError(
            f"chat.create() got unexpected keyword argument(s): {sorted(unknown)}. "
            f"Allowed: {sorted(_ALLOWED_CHAT_PARAMS)}."
        )
    spec = get_model_spec(model)
    if spec is not None:
        extra = spec.apply_defaults(extra)
        spec.validate_params(extra)
    if default_cache_key is not None and "prompt_cache_key" not in extra:
        extra = {**extra, "prompt_cache_key": default_cache_key}
    if extra.get("tools"):
        extra = {**extra, "tools": [_coerce_tool(t) for t in extra["tools"]]}
    payload: dict[str, Any] = {
        "model": str(model),
        "messages": list(messages),
        **extra,
    }
    if stream:
        payload["stream"] = True
    req = ChatCompletionRequest.model_validate(payload)
    return req.model_dump(exclude_none=True, by_alias=True, mode="json")


class Chat:
    def __init__(self, client: KimiClient) -> None:
        self._client = client

    @overload
    def create(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        stream: Literal[False] = False,
        **params: Any,
    ) -> ChatCompletion: ...

    @overload
    def create(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        stream: Literal[True],
        **params: Any,
    ) -> Iterator[ChatCompletionChunk]: ...

    def create(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        stream: bool = False,
        **params: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        messages = list(messages)
        _enforce_single_turn(messages)
        return self._create(model=model, messages=messages, stream=stream, **params)

    def _create(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        stream: bool = False,
        **params: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Internal: same as :meth:`create` but skips the manual-multi-turn
        gate. Used by :class:`Session` and :func:`run_tools`, which manage
        history correctly. Not part of the public API."""
        body = _build_request_body(
            model=model,
            messages=messages,
            stream=stream,
            extra=params,
            default_cache_key=self._client._prompt_cache_key,
        )
        if stream:
            return self._stream(body, model=model)
        response = self._client._request("POST", "/chat/completions", json=body)
        completion = ChatCompletion.model_validate(response.json())
        self._client.cache_stats.record(completion.usage, model=model)
        return completion

    def _stream(
        self, body: dict[str, Any], *, model: Model | str
    ) -> Iterator[ChatCompletionChunk]:
        def _open() -> tuple[ExitStack, httpx.Response]:
            stack = ExitStack()
            try:
                response = stack.enter_context(
                    self._client._http.stream(
                        "POST",
                        "/chat/completions",
                        json=body,
                        headers=self._client._auth,
                    )
                )
                if response.status_code >= 400:
                    response.read()
                    raise_for_status(response)
            except BaseException:
                stack.close()
                raise
            return stack, response

        stack, response = retry_sync(self._client._retry, _open)
        with stack:
            for line in response.iter_lines():
                chunk = parse_sse_line(line)
                if chunk is None:
                    continue
                parsed = ChatCompletionChunk.model_validate(chunk)
                self._client.cache_stats.record(parsed.usage, model=model)
                yield parsed

    def prefill(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        prefill: str,
        **params: Any,
    ) -> PrefilledCompletion:
        """Send a partial-mode request and get back ``prefill + new tokens``.

        ``messages`` should *not* include the assistant prefill — this helper
        appends ``{"role": "assistant", "content": prefill, "partial": True}``
        for you and concatenates the response onto the prefill in
        ``PrefilledCompletion.text``.
        """
        msgs = list(messages) + [
            Message(role=Role.ASSISTANT, content=prefill, partial=True)
        ]
        completion = self._create(model=model, messages=msgs, **params)
        assert isinstance(completion, ChatCompletion)
        body = completion.choices[0].message.content or ""
        return PrefilledCompletion(completion=completion, text=prefill + body)

    def parse(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        response_format: type[T] | TypeAdapter[T] | Any,
        **params: Any,
    ) -> ParsedChatCompletion[T]:
        """Constrain output to a JSON schema and return a parsed value.

        ``response_format`` accepts a Pydantic ``BaseModel`` subclass, a
        ``TypeAdapter`` (e.g. ``TypeAdapter(list[int])``), or any type
        ``TypeAdapter`` can wrap. The schema is sent as
        ``response_format={"type":"json_schema","json_schema":{...}}``.
        """
        messages = list(messages)
        _enforce_single_turn(messages)
        schema, name, parser = _resolve_schema(response_format)
        rf = {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": schema, "strict": True},
        }
        completion = self._create(
            model=model, messages=messages, response_format=rf, **params
        )
        assert isinstance(completion, ChatCompletion)
        return _parse_completion(completion, parser)

    def stream_events(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        **params: Any,
    ) -> Iterator[Any]:
        """Stream the response as typed events (TextDelta, ReasoningDelta, …).

        Convenience wrapper around ``create(stream=True, ...)`` +
        :func:`kimi_agents_python.stream_events`.
        """
        from ..events import stream_events

        messages = list(messages)
        _enforce_single_turn(messages)
        chunks = self._create(model=model, messages=messages, stream=True, **params)
        assert not isinstance(chunks, ChatCompletion)
        yield from stream_events(chunks)

    def stream_with_reconnect(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        max_attempts: int = 5,
        retry_delay: float = 1.0,
        **params: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Stream chunks with prefill-based resumption on transport errors.

        On ``httpx.RemoteProtocolError`` / ``ReadError`` / ``ReadTimeout`` the
        wrapper restarts the request with the accumulated text appended as a
        partial-mode assistant prefill. Tool-call streams cannot be safely
        resumed (partial JSON arguments would corrupt on replay): if any
        tool-call delta has already been emitted when the drop happens, the
        original transport error is propagated.
        """
        messages = list(messages)
        _enforce_single_turn(messages)
        content_parts: list[str] = []
        saw_tool_call = False
        for attempt in range(max_attempts):
            try:
                resume = (
                    messages + [
                        Message(
                            role=Role.ASSISTANT,
                            content="".join(content_parts),
                            partial=True,
                        )
                    ]
                    if content_parts
                    else messages
                )
                stream = self._create(
                    model=model, messages=resume, stream=True, **params
                )
                assert not isinstance(stream, ChatCompletion)
                for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        content_parts.append(delta.content)
                    if delta and delta.tool_calls:
                        saw_tool_call = True
                    yield chunk
                return
            except (
                httpx.RemoteProtocolError,
                httpx.ReadError,
                httpx.ReadTimeout,
            ):
                if saw_tool_call:
                    raise
                if attempt + 1 == max_attempts:
                    raise
                time.sleep(retry_delay)


class AsyncChat:
    def __init__(self, client: AsyncKimiClient) -> None:
        self._client = client

    @overload
    async def create(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        stream: Literal[False] = False,
        **params: Any,
    ) -> ChatCompletion: ...

    @overload
    async def create(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        stream: Literal[True],
        **params: Any,
    ) -> AsyncIterator[ChatCompletionChunk]: ...

    async def create(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        stream: bool = False,
        **params: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        messages = list(messages)
        _enforce_single_turn(messages)
        return await self._create(
            model=model, messages=messages, stream=stream, **params
        )

    async def _create(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        stream: bool = False,
        **params: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        body = _build_request_body(
            model=model,
            messages=messages,
            stream=stream,
            extra=params,
            default_cache_key=self._client._prompt_cache_key,
        )
        if stream:
            return self._stream(body, model=model)
        response = await self._client._request(
            "POST", "/chat/completions", json=body
        )
        completion = ChatCompletion.model_validate(response.json())
        self._client.cache_stats.record(completion.usage, model=model)
        return completion

    async def _stream(
        self, body: dict[str, Any], *, model: Model | str
    ) -> AsyncIterator[ChatCompletionChunk]:
        async def _open() -> tuple[AsyncExitStack, httpx.Response]:
            stack = AsyncExitStack()
            try:
                response = await stack.enter_async_context(
                    self._client._http.stream(
                        "POST",
                        "/chat/completions",
                        json=body,
                        headers=self._client._auth,
                    )
                )
                if response.status_code >= 400:
                    await response.aread()
                    raise_for_status(response)
            except BaseException:
                await stack.aclose()
                raise
            return stack, response

        stack, response = await retry_async(self._client._retry, _open)
        async with stack:
            async for line in response.aiter_lines():
                chunk = parse_sse_line(line)
                if chunk is None:
                    continue
                parsed = ChatCompletionChunk.model_validate(chunk)
                self._client.cache_stats.record(parsed.usage, model=model)
                yield parsed

    async def prefill(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        prefill: str,
        **params: Any,
    ) -> PrefilledCompletion:
        """Async counterpart to :meth:`Chat.prefill`."""
        msgs = list(messages) + [
            Message(role=Role.ASSISTANT, content=prefill, partial=True)
        ]
        completion = await self._create(model=model, messages=msgs, **params)
        assert isinstance(completion, ChatCompletion)
        body = completion.choices[0].message.content or ""
        return PrefilledCompletion(completion=completion, text=prefill + body)

    async def parse(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        response_format: type[T] | TypeAdapter[T] | Any,
        **params: Any,
    ) -> ParsedChatCompletion[T]:
        """Async counterpart to :meth:`Chat.parse`."""
        messages = list(messages)
        _enforce_single_turn(messages)
        schema, name, parser = _resolve_schema(response_format)
        rf = {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": schema, "strict": True},
        }
        completion = await self._create(
            model=model, messages=messages, response_format=rf, **params
        )
        assert isinstance(completion, ChatCompletion)
        return _parse_completion(completion, parser)

    async def stream_events(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        **params: Any,
    ) -> AsyncIterator[Any]:
        """Async counterpart to :meth:`Chat.stream_events`."""
        from ..events import astream_events

        messages = list(messages)
        _enforce_single_turn(messages)
        chunks = await self._create(
            model=model, messages=messages, stream=True, **params
        )
        assert not isinstance(chunks, ChatCompletion)
        async for ev in astream_events(chunks):
            yield ev

    async def stream_with_reconnect(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        max_attempts: int = 5,
        retry_delay: float = 1.0,
        **params: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Async counterpart to :meth:`Chat.stream_with_reconnect`."""
        messages = list(messages)
        _enforce_single_turn(messages)
        content_parts: list[str] = []
        saw_tool_call = False
        for attempt in range(max_attempts):
            try:
                resume = (
                    messages + [
                        Message(
                            role=Role.ASSISTANT,
                            content="".join(content_parts),
                            partial=True,
                        )
                    ]
                    if content_parts
                    else messages
                )
                stream = await self._create(
                    model=model, messages=resume, stream=True, **params
                )
                assert not isinstance(stream, ChatCompletion)
                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        content_parts.append(delta.content)
                    if delta and delta.tool_calls:
                        saw_tool_call = True
                    yield chunk
                return
            except (
                httpx.RemoteProtocolError,
                httpx.ReadError,
                httpx.ReadTimeout,
            ):
                if saw_tool_call:
                    raise
                if attempt + 1 == max_attempts:
                    raise
                await asyncio.sleep(retry_delay)
