from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator
from contextlib import AsyncExitStack, ExitStack
from typing import TYPE_CHECKING, Any, Literal, overload

import httpx

from .._enums import Model, Role
from .._http import parse_sse_line, raise_for_status
from .._retry import retry_async, retry_sync
from ..exceptions import ManualMultiTurnError
from ..specs import get_model_spec
from ..tools import KimiTool
from ..types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Message,
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
        spec.validate_params(extra)
    if default_cache_key is not None and "prompt_cache_key" not in extra:
        extra = {**extra, "prompt_cache_key": default_cache_key}
    if extra.get("tools"):
        extra = {
            **extra,
            "tools": [
                t.to_tool_def() if isinstance(t, KimiTool) else t
                for t in extra["tools"]
            ],
        }
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
            return self._stream(body)
        response = self._client._request("POST", "/chat/completions", json=body)
        completion = ChatCompletion.model_validate(response.json())
        self._client.cache_stats.record(completion.usage)
        return completion

    def _stream(self, body: dict[str, Any]) -> Iterator[ChatCompletionChunk]:
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
                self._client.cache_stats.record(parsed.usage)
                yield parsed


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
            return self._stream(body)
        response = await self._client._request(
            "POST", "/chat/completions", json=body
        )
        completion = ChatCompletion.model_validate(response.json())
        self._client.cache_stats.record(completion.usage)
        return completion

    async def _stream(
        self, body: dict[str, Any]
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
                self._client.cache_stats.record(parsed.usage)
                yield parsed
