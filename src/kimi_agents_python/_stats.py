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
