from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator
from types import TracebackType
from typing import Any, ClassVar, Literal, overload

import httpx

from ._enums import AVAILABLE_MODELS, Model
from ._http import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    bearer_headers,
    parse_sse_line,
    raise_for_status,
    resolve_api_key,
)
from ._retry import DEFAULT_RETRY, RetryConfig, retry_async, retry_sync
from .specs import get_model_spec
from .tools import KimiTool
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


def _resolve_retry(
    max_retries: int | None, retry_config: RetryConfig | None
) -> RetryConfig:
    if retry_config is not None:
        return retry_config
    if max_retries is None:
        return DEFAULT_RETRY
    return RetryConfig(max_retries=max_retries)


def _build_request_body(
    *,
    model: Model | str,
    messages: Iterable[_MessageInput],
    stream: bool,
    extra: dict[str, Any],
) -> dict[str, Any]:
    spec = get_model_spec(model)
    if spec is not None:
        spec.validate_params(extra)
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
        max_retries: int | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._api_key = resolve_api_key(api_key)
        self._base_url = base_url.rstrip("/")
        self._auth = bearer_headers(self._api_key)
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
        )
        self._retry = _resolve_retry(max_retries, retry_config)

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
        stream: Literal[False] = False,
        **params: Any,
    ) -> ChatCompletion: ...

    @overload
    def chat(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        stream: Literal[True],
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

        def _do() -> ChatCompletion:
            response = self._http.post(
                "/chat/completions", json=body, headers=self._auth
            )
            raise_for_status(response)
            return ChatCompletion.model_validate(response.json())

        return retry_sync(self._retry, _do)

    def _stream_chat(self, body: dict[str, Any]) -> Iterator[ChatCompletionChunk]:
        def _open() -> tuple[Any, httpx.Response]:
            cm = self._http.stream(
                "POST", "/chat/completions", json=body, headers=self._auth
            )
            response = cm.__enter__()
            try:
                if response.status_code >= 400:
                    response.read()
                    raise_for_status(response)
            except BaseException:
                cm.__exit__(None, None, None)
                raise
            return cm, response

        cm, response = retry_sync(self._retry, _open)
        try:
            for line in response.iter_lines():
                chunk = parse_sse_line(line)
                if chunk is None:
                    continue
                yield ChatCompletionChunk.model_validate(chunk)
        finally:
            cm.__exit__(None, None, None)

    def list_models(self) -> list[ModelInfo]:
        def _do() -> list[ModelInfo]:
            response = self._http.get("/models", headers=self._auth)
            raise_for_status(response)
            return ModelList.model_validate(response.json()).data

        return retry_sync(self._retry, _do)

    def estimate_tokens(
        self, *, model: Model | str, messages: Iterable[_MessageInput]
    ) -> TokenEstimate:
        body = {"model": str(model), "messages": list(messages)}

        def _do() -> TokenEstimate:
            response = self._http.post(
                "/tokenizers/estimate-token-count", json=body, headers=self._auth
            )
            raise_for_status(response)
            return TokenEstimate.model_validate(response.json())

        return retry_sync(self._retry, _do)

    def balance(self) -> BalanceInfo:
        def _do() -> BalanceInfo:
            response = self._http.get("/users/me/balance", headers=self._auth)
            raise_for_status(response)
            return BalanceInfo.model_validate(response.json())

        return retry_sync(self._retry, _do)


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
        max_retries: int | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._api_key = resolve_api_key(api_key)
        self._base_url = base_url.rstrip("/")
        self._auth = bearer_headers(self._api_key)
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
        )
        self._retry = _resolve_retry(max_retries, retry_config)

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

    @overload
    async def chat(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        stream: Literal[False] = False,
        **params: Any,
    ) -> ChatCompletion: ...

    @overload
    async def chat(
        self,
        *,
        model: Model | str,
        messages: Iterable[_MessageInput],
        stream: Literal[True],
        **params: Any,
    ) -> AsyncIterator[ChatCompletionChunk]: ...

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

        async def _do() -> ChatCompletion:
            response = await self._http.post(
                "/chat/completions", json=body, headers=self._auth
            )
            raise_for_status(response)
            return ChatCompletion.model_validate(response.json())

        return await retry_async(self._retry, _do)

    async def _stream_chat(
        self, body: dict[str, Any]
    ) -> AsyncIterator[ChatCompletionChunk]:
        async def _open() -> tuple[Any, httpx.Response]:
            cm = self._http.stream(
                "POST", "/chat/completions", json=body, headers=self._auth
            )
            response = await cm.__aenter__()
            try:
                if response.status_code >= 400:
                    await response.aread()
                    raise_for_status(response)
            except BaseException:
                await cm.__aexit__(None, None, None)
                raise
            return cm, response

        cm, response = await retry_async(self._retry, _open)
        try:
            async for line in response.aiter_lines():
                chunk = parse_sse_line(line)
                if chunk is None:
                    continue
                yield ChatCompletionChunk.model_validate(chunk)
        finally:
            await cm.__aexit__(None, None, None)

    async def list_models(self) -> list[ModelInfo]:
        async def _do() -> list[ModelInfo]:
            response = await self._http.get("/models", headers=self._auth)
            raise_for_status(response)
            return ModelList.model_validate(response.json()).data

        return await retry_async(self._retry, _do)

    async def estimate_tokens(
        self, *, model: Model | str, messages: Iterable[_MessageInput]
    ) -> TokenEstimate:
        body = {"model": str(model), "messages": list(messages)}

        async def _do() -> TokenEstimate:
            response = await self._http.post(
                "/tokenizers/estimate-token-count", json=body, headers=self._auth
            )
            raise_for_status(response)
            return TokenEstimate.model_validate(response.json())

        return await retry_async(self._retry, _do)

    async def balance(self) -> BalanceInfo:
        async def _do() -> BalanceInfo:
            response = await self._http.get("/users/me/balance", headers=self._auth)
            raise_for_status(response)
            return BalanceInfo.model_validate(response.json())

        return await retry_async(self._retry, _do)
