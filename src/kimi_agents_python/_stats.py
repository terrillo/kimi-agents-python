from __future__ import annotations

from dataclasses import dataclass

from .types import Usage


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

    def record(self, usage: Usage) -> None:
        self.requests += 1
        self.prompt_tokens += usage.prompt_tokens
        nested = (
            usage.prompt_tokens_details.cached_tokens
            if usage.prompt_tokens_details is not None
            else None
        )
        self.cached_tokens += nested if nested is not None else (usage.cached_tokens or 0)

    def reset(self) -> None:
        self.requests = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
