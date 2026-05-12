"""Per-model pricing and cost calculation.

Per-1M-token prices and the fixed ``$web_search`` per-call fee, copied from
``.platform/pricing/``. ``moonshot-v1-*`` models have no cache tier — both
cache-hit and cache-miss columns use the published input price.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ._enums import Model
from .types import Usage, _cached_tokens_from


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """USD per 1M tokens, broken out by cache state."""

    input_cache_hit_per_1m: Decimal
    input_cache_miss_per_1m: Decimal
    output_per_1m: Decimal


_PER_MILLION = Decimal("1000000")


MODEL_PRICING: dict[Model, ModelPricing] = {
    Model.KIMI_K2_6: ModelPricing(
        Decimal("0.16"), Decimal("0.95"), Decimal("4.00")
    ),
    Model.KIMI_K2_5: ModelPricing(
        Decimal("0.16"), Decimal("0.95"), Decimal("4.00")
    ),
    Model.KIMI_K2_0905_PREVIEW: ModelPricing(
        Decimal("0.15"), Decimal("0.60"), Decimal("2.50")
    ),
    Model.KIMI_K2_0711_PREVIEW: ModelPricing(
        Decimal("0.15"), Decimal("0.60"), Decimal("2.50")
    ),
    Model.KIMI_K2_TURBO_PREVIEW: ModelPricing(
        Decimal("0.15"), Decimal("1.15"), Decimal("8.00")
    ),
    Model.KIMI_K2_THINKING: ModelPricing(
        Decimal("0.15"), Decimal("0.60"), Decimal("2.50")
    ),
    Model.KIMI_K2_THINKING_TURBO: ModelPricing(
        Decimal("0.15"), Decimal("1.15"), Decimal("8.00")
    ),
    Model.MOONSHOT_V1_8K: ModelPricing(
        Decimal("0.20"), Decimal("0.20"), Decimal("2.00")
    ),
    Model.MOONSHOT_V1_32K: ModelPricing(
        Decimal("1.00"), Decimal("1.00"), Decimal("3.00")
    ),
    Model.MOONSHOT_V1_128K: ModelPricing(
        Decimal("2.00"), Decimal("2.00"), Decimal("5.00")
    ),
    Model.MOONSHOT_V1_8K_VISION_PREVIEW: ModelPricing(
        Decimal("0.20"), Decimal("0.20"), Decimal("2.00")
    ),
    Model.MOONSHOT_V1_32K_VISION_PREVIEW: ModelPricing(
        Decimal("1.00"), Decimal("1.00"), Decimal("3.00")
    ),
    Model.MOONSHOT_V1_128K_VISION_PREVIEW: ModelPricing(
        Decimal("2.00"), Decimal("2.00"), Decimal("5.00")
    ),
    # MOONSHOT_V1_AUTO has no fixed price (billing depends on routed model).
}


WEB_SEARCH_CALL_USD: Decimal = Decimal("0.005")
"""Per-call fee for ``$web_search`` (``.platform/pricing/tools.md``)."""


def _resolve_model(model: Model | str) -> Model | None:
    if isinstance(model, Model):
        return model
    try:
        return Model(model)
    except ValueError:
        return None


def usage_to_cost(usage: Usage | None, model: Model | str) -> Decimal:
    """USD cost for one chat completion, splitting cache-hit and cache-miss inputs.

    Returns ``Decimal("0")`` for unknown / un-priced models (e.g.
    ``moonshot-v1-auto``) or ``None`` usage, so callers can accumulate without
    branching on model support.
    """
    if usage is None:
        return Decimal("0")
    resolved = _resolve_model(model)
    if resolved is None:
        return Decimal("0")
    pricing = MODEL_PRICING.get(resolved)
    if pricing is None:
        return Decimal("0")
    cached = _cached_tokens_from(usage)
    uncached = max(usage.prompt_tokens - cached, 0)
    raw = (
        Decimal(cached) * pricing.input_cache_hit_per_1m
        + Decimal(uncached) * pricing.input_cache_miss_per_1m
        + Decimal(usage.completion_tokens) * pricing.output_per_1m
    )
    return raw / _PER_MILLION


__all__ = ["MODEL_PRICING", "ModelPricing", "WEB_SEARCH_CALL_USD", "usage_to_cost"]
