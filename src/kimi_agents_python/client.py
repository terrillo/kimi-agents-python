from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator
from types import TracebackType
from typing import Any, ClassVar, overload

import httpx

from ._enums import AVAILABLE_MODELS, Model
from ._http import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    auth_headers,
    parse_sse_line,
    raise_for_status,
    resolve_api_key,
)
from .types import (
    BalanceInfo,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Message,
    ModelInfo,
    ModelList,
    TokenEstimate,
)

_MessageInput = Message | dict[str, Any]


def _build_request_body(
    *,
    model: Model | str,
    messages: Iterable[_MessageInput],
    stream: bool,
    extra: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": str(model),
        "messages": list(messages),
        **extra,
    }
    if stream:
        payload["stream"] = True
    req = ChatCompletionRequest.model_validate(payload)
    return req.model_dump(exclude_none=True, by_alias=True, mode="json")


class KimiClient:
    """Synchronous Kimi (Moonshot) chat client."""

    AVAILABLE_MODELS: ClassVar[tuple[Model, ...]] = AVAILABLE_MODELS
    Model: ClassVar[type[Model]] = Model

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = resolve_api_key(api_key)
        self._base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
            headers=auth_headers(self._api_key),
        )
        if not self._owns_client:
            self._http.headers.update(auth_headers(self._api_key))

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> KimiClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @overload
    def chat(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        stream: bool = False,
        **params: Any,
    ) -> ChatCompletion: ...

    @overload
    def chat(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        stream: bool,
        **params: Any,
    ) -> Iterator[ChatCompletionChunk]: ...

    def chat(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        stream: bool = False,
        **params: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        body = _build_request_body(
            model=model, messages=messages, stream=stream, extra=params
        )
        if stream:
            return self._stream_chat(body)
        response = self._http.post("/chat/completions", json=body)
        raise_for_status(response)
        return ChatCompletion.model_validate(response.json())

    def _stream_chat(self, body: dict[str, Any]) -> Iterator[ChatCompletionChunk]:
        with self._http.stream("POST", "/chat/completions", json=body) as response:
            if response.status_code >= 400:
                response.read()
                raise_for_status(response)
            for line in response.iter_lines():
                chunk = parse_sse_line(line)
                if chunk is None:
                    continue
                yield ChatCompletionChunk.model_validate(chunk)

    def list_models(self) -> list[ModelInfo]:
        response = self._http.get("/models")
        raise_for_status(response)
        return ModelList.model_validate(response.json()).data

    def estimate_tokens(
        self, *, model: Model | str, messages: Iterable[_MessageInput]
    ) -> TokenEstimate:
        body = {"model": str(model), "messages": list(messages)}
        response = self._http.post("/tokenizers/estimate-token-count", json=body)
        raise_for_status(response)
        return TokenEstimate.model_validate(response.json())

    def balance(self) -> BalanceInfo:
        response = self._http.get("/users/me/balance")
        raise_for_status(response)
        return BalanceInfo.model_validate(response.json())


class AsyncKimiClient:
    """Asynchronous Kimi (Moonshot) chat client."""

    AVAILABLE_MODELS: ClassVar[tuple[Model, ...]] = AVAILABLE_MODELS
    Model: ClassVar[type[Model]] = Model

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = resolve_api_key(api_key)
        self._base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers=auth_headers(self._api_key),
        )
        if not self._owns_client:
            self._http.headers.update(auth_headers(self._api_key))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncKimiClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def chat(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        stream: bool = False,
        **params: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        body = _build_request_body(
            model=model, messages=messages, stream=stream, extra=params
        )
        if stream:
            return self._stream_chat(body)
        response = await self._http.post("/chat/completions", json=body)
        raise_for_status(response)
        return ChatCompletion.model_validate(response.json())

    async def _stream_chat(
        self, body: dict[str, Any]
    ) -> AsyncIterator[ChatCompletionChunk]:
        async with self._http.stream("POST", "/chat/completions", json=body) as response:
            if response.status_code >= 400:
                await response.aread()
                raise_for_status(response)
            async for line in response.aiter_lines():
                chunk = parse_sse_line(line)
                if chunk is None:
                    continue
                yield ChatCompletionChunk.model_validate(chunk)

    async def list_models(self) -> list[ModelInfo]:
        response = await self._http.get("/models")
        raise_for_status(response)
        return ModelList.model_validate(response.json()).data

    async def estimate_tokens(
        self, *, model: Model | str, messages: Iterable[_MessageInput]
    ) -> TokenEstimate:
        body = {"model": str(model), "messages": list(messages)}
        response = await self._http.post("/tokenizers/estimate-token-count", json=body)
        raise_for_status(response)
        return TokenEstimate.model_validate(response.json())

    async def balance(self) -> BalanceInfo:
        response = await self._http.get("/users/me/balance")
        raise_for_status(response)
        return BalanceInfo.model_validate(response.json())
