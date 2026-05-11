from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

import httpx

from .exceptions import KimiAPIError

T = TypeVar("T")

_RETRYABLE_STATUS: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class RetryConfig:
    """Auto-retry policy for transient API failures.

    Retries on HTTP 429 (rate limit / engine overloaded), 5xx server errors,
    and network-level errors from httpx. Honors a numeric ``Retry-After``
    response header when present.
    """

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: float = 0.25


DEFAULT_RETRY = RetryConfig()
NO_RETRY = RetryConfig(max_retries=0)


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, KimiAPIError):
        return exc.status_code in _RETRYABLE_STATUS
    if isinstance(exc, (httpx.TransportError, httpx.RemoteProtocolError)):
        return True
    return False


def _retry_after(exc: BaseException) -> float | None:
    if isinstance(exc, KimiAPIError):
        return exc.retry_after
    return None


def _compute_delay(
    attempt: int, cfg: RetryConfig, retry_after: float | None
) -> float:
    if retry_after is not None and retry_after >= 0:
        return min(retry_after, cfg.max_delay)
    base = cfg.initial_delay * (cfg.backoff_factor ** max(0, attempt - 1))
    base = min(base, cfg.max_delay)
    if cfg.jitter > 0:
        base = base * (1 + cfg.jitter * (2 * random.random() - 1))
    return max(0.0, base)


def retry_sync(cfg: RetryConfig, fn: Callable[[], T]) -> T:
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= cfg.max_retries or not _should_retry(exc):
                raise
            attempt += 1
            time.sleep(_compute_delay(attempt, cfg, _retry_after(exc)))


async def retry_async(cfg: RetryConfig, fn: Callable[[], Awaitable[T]]) -> T:
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:
            if attempt >= cfg.max_retries or not _should_retry(exc):
                raise
            attempt += 1
            await asyncio.sleep(_compute_delay(attempt, cfg, _retry_after(exc)))
