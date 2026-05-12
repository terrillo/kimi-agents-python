from __future__ import annotations

from decimal import Decimal

from kimi_agents_python import (
    CacheStats,
    MODEL_PRICING,
    Model,
    TokenStats,
    Usage,
    usage_to_cost,
)


def test_pricing_table_covers_active_models() -> None:
    for m in (
        Model.KIMI_K2_6,
        Model.KIMI_K2_THINKING,
        Model.KIMI_K2_TURBO_PREVIEW,
        Model.MOONSHOT_V1_128K,
    ):
        assert m in MODEL_PRICING


def test_usage_to_cost_splits_cache_hit_and_miss() -> None:
    usage = Usage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        total_tokens=2_000_000,
        prompt_tokens_details={"cached_tokens": 400_000},
    )
    cost = usage_to_cost(usage, Model.KIMI_K2_6)
    # 0.4M @ 0.16 + 0.6M @ 0.95 + 1M @ 4.00 = 0.064 + 0.570 + 4.000 = 4.634
    assert cost == Decimal("4.634")


def test_usage_to_cost_unknown_model_is_zero() -> None:
    usage = Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    assert usage_to_cost(usage, "totally-fake-model") == Decimal("0")
    assert usage_to_cost(None, Model.KIMI_K2_6) == Decimal("0")


def test_cache_stats_record_accumulates_cost_when_model_given() -> None:
    usage = Usage(
        prompt_tokens=1_000_000,
        completion_tokens=0,
        total_tokens=1_000_000,
        prompt_tokens_details={"cached_tokens": 0},
    )
    stats = CacheStats()
    stats.record(usage, model=Model.KIMI_K2_6)
    assert stats.cost_usd == Decimal("0.95")  # all cache-miss input
    # Same record without model: counters update, cost does not.
    stats2 = CacheStats()
    stats2.record(usage)
    assert stats2.cost_usd == Decimal("0")
    assert stats2.prompt_tokens == 1_000_000


def test_token_stats_record_accumulates_cost() -> None:
    usage = Usage(prompt_tokens=0, completion_tokens=1_000_000, total_tokens=1_000_000)
    stats = TokenStats()
    stats.record(usage, model=Model.KIMI_K2_6)
    assert stats.cost_usd == Decimal("4.00")
    stats.reset()
    assert stats.cost_usd == Decimal("0")
