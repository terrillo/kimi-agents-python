from __future__ import annotations

import uuid
from types import TracebackType
from typing import Any, ClassVar

import httpx

from ._enums import AVAILABLE_MODELS, Model
from ._http import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    bearer_headers,
    raise_for_status,
    resolve_api_key,
)
from ._retry import DEFAULT_RETRY, RetryConfig, retry_async, retry_sync
from ._stats import CacheStats
from .resources import (
    Account,
    AsyncAccount,
    AsyncBatches,
    AsyncChat,
    AsyncFiles,
    AsyncFormulas,
    AsyncModels,
    AsyncTokenizers,
    Batches,
    Chat,
    Files,
    Formulas,
    Models,
    Tokenizers,
)


def _resolve_retry(retries: int | RetryConfig | None) -> RetryConfig:
    if retries is None:
        return DEFAULT_RETRY
    if isinstance(retries, RetryConfig):
        return retries
    return RetryConfig(max_retries=retries)


class KimiClient:
    """Synchronous Kimi (Moonshot) API client.

    Resources are exposed as namespaces:

    * ``client.chat.create(...)`` — chat completions
    * ``client.files.create/list/retrieve/delete/content`` — file management
    * ``client.batches.create/retrieve/list/cancel`` — batch inference
    * ``client.models.list()`` — available models
    * ``client.tokenizers.estimate(...)`` — token-count estimate
    * ``client.account.balance()`` — billing balance
    * ``client.formulas.tools/invoke/load`` — official Formula API tools
    """

    AVAILABLE_MODELS: ClassVar[tuple[Model, ...]] = AVAILABLE_MODELS
    Model: ClassVar[type[Model]] = Model

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.Client | None = None,
        retries: int | RetryConfig | None = None,
        prompt_cache_key: str | None = None,
    ) -> None:
        self._api_key = resolve_api_key(api_key)
        self._base_url = base_url.rstrip("/")
        self._auth = bearer_headers(self._api_key)
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
        )
        self._retry = _resolve_retry(retries)
        self._prompt_cache_key = prompt_cache_key
        self.cache_stats = CacheStats()

        self.chat = Chat(self)
        self.files = Files(self)
        self.batches = Batches(self)
        self.models = Models(self)
        self.tokenizers = Tokenizers(self)
        self.account = Account(self)
        self.formulas = Formulas(self)

    @property
    def base_url(self) -> str:
        return self._base_url

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        def _do() -> httpx.Response:
            response = self._http.request(
                method, path, headers=self._auth, **kwargs
            )
            raise_for_status(response)
            return response

        return retry_sync(self._retry, _do)

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

    @classmethod
    def with_moonpalace(
        cls,
        *,
        port: int = 9988,
        host: str = "127.0.0.1",
        trace_id: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> KimiClient:
        """Build a client pointed at a local MoonPalace debugging proxy.

        Sets ``base_url`` to ``http://{host}:{port}/v1`` and stamps a
        per-client ``X-Msh-Trace-Id`` header (one UUID for the lifetime of the
        client) so every request through this client is grouped under the same
        trace id in ``moonpalace list``.

        Start MoonPalace separately: ``moonpalace start --port 9988``.
        """
        client = cls(api_key=api_key, base_url=f"http://{host}:{port}/v1", **kwargs)
        client._auth = {
            **client._auth,
            "X-Msh-Trace-Id": trace_id or uuid.uuid4().hex,
        }
        return client


class AsyncKimiClient:
    """Asynchronous Kimi (Moonshot) API client.

    Mirrors :class:`KimiClient`; each resource method is a coroutine.
    """

    AVAILABLE_MODELS: ClassVar[tuple[Model, ...]] = AVAILABLE_MODELS
    Model: ClassVar[type[Model]] = Model

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.AsyncClient | None = None,
        retries: int | RetryConfig | None = None,
        prompt_cache_key: str | None = None,
    ) -> None:
        self._api_key = resolve_api_key(api_key)
        self._base_url = base_url.rstrip("/")
        self._auth = bearer_headers(self._api_key)
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
        )
        self._retry = _resolve_retry(retries)
        self._prompt_cache_key = prompt_cache_key
        self.cache_stats = CacheStats()

        self.chat = AsyncChat(self)
        self.files = AsyncFiles(self)
        self.batches = AsyncBatches(self)
        self.models = AsyncModels(self)
        self.tokenizers = AsyncTokenizers(self)
        self.account = AsyncAccount(self)
        self.formulas = AsyncFormulas(self)

    @property
    def base_url(self) -> str:
        return self._base_url

    async def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> httpx.Response:
        async def _do() -> httpx.Response:
            response = await self._http.request(
                method, path, headers=self._auth, **kwargs
            )
            raise_for_status(response)
            return response

        return await retry_async(self._retry, _do)

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

    @classmethod
    def with_moonpalace(
        cls,
        *,
        port: int = 9988,
        host: str = "127.0.0.1",
        trace_id: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> AsyncKimiClient:
        """Async counterpart to :meth:`KimiClient.with_moonpalace`."""
        client = cls(api_key=api_key, base_url=f"http://{host}:{port}/v1", **kwargs)
        client._auth = {
            **client._auth,
            "X-Msh-Trace-Id": trace_id or uuid.uuid4().hex,
        }
        return client


__all__ = ["AsyncKimiClient", "KimiClient"]
