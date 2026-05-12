from __future__ import annotations

from dataclasses import dataclass

from .types import Usage, _cached_tokens_from


@dataclass(slots=True)
class CacheStats:
    """Cumulative prompt-cache hit statistics across chat completions on one client.

    Counters accumulate over every successful chat call (streaming included when
    ``stream_options={"include_usage": True}``). ``cached_tokens`` prefers the
    nested ``usage.prompt_tokens_details.cached_tokens`` field and falls back to
    the legacy top-level ``usage.cached_tokens`` for older payloads.
    """

    requests: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0

    @property
    def hit_ratio(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def record(self, usage: Usage | None) -> None:
        """Accumulate one chat usage record; ``None`` is a no-op."""
        if usage is None:
            return
        self.requests += 1
        self.prompt_tokens += usage.prompt_tokens
        self.cached_tokens += _cached_tokens_from(usage)

    def reset(self) -> None:
        self.requests = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0


@dataclass(slots=True)
class TokenStats:
    """Cumulative token-usage counters for a single conversation.

    Sibling to :class:`CacheStats` but tracks the full breakdown
    (prompt / completion / total / cached) rather than just cache hits.
    Used by :class:`~kimi_agents_python.session.Session` to expose per-session
    totals independent of the client-wide :attr:`CacheStats`.
    """

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def record(self, usage: Usage | None) -> None:
        if usage is None:
            return
        self.requests += 1
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.cached_tokens += _cached_tokens_from(usage)

    def reset(self) -> None:
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cached_tokens = 0
