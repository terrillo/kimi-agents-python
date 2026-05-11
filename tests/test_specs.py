from __future__ import annotations

import pytest

from kimi_agents_python import (
    MODEL_SPECS,
    Model,
    ModelSpec,
    ThinkingIncompatibilityError,
    get_model_spec,
)


def test_every_model_enum_value_has_a_spec() -> None:
    missing = [m for m in Model if m not in MODEL_SPECS]
    assert missing == []


def test_get_model_spec_accepts_enum_and_string() -> None:
    spec = get_model_spec(Model.KIMI_K2_6)
    assert isinstance(spec, ModelSpec)
    assert spec is get_model_spec("kimi-k2.6")


def test_get_model_spec_returns_none_for_unknown() -> None:
    assert get_model_spec("kimi-k2.99-future") is None


def test_k26_locks_temperature() -> None:
    spec = MODEL_SPECS[Model.KIMI_K2_6]
    spec.validate_params({"temperature": 1.0})  # equal to default → fine
    with pytest.raises(ValueError, match="temperature is locked"):
        spec.validate_params({"temperature": 0.3})


def test_k26_locks_top_p_and_n_and_penalty() -> None:
    spec = MODEL_SPECS[Model.KIMI_K2_6]
    with pytest.raises(ValueError, match="top_p is locked"):
        spec.validate_params({"top_p": 0.8})
    with pytest.raises(ValueError, match="n is locked"):
        spec.validate_params({"n": 2})
    with pytest.raises(ValueError, match="presence_penalty is locked"):
        spec.validate_params({"presence_penalty": 0.5})
    with pytest.raises(ValueError, match="frequency_penalty is locked"):
        spec.validate_params({"frequency_penalty": 0.5})


def test_k26_thinking_is_configurable() -> None:
    spec = MODEL_SPECS[Model.KIMI_K2_6]
    spec.validate_params({"thinking": {"type": "enabled"}})
    spec.validate_params({"thinking": {"type": "disabled"}})


def test_k2_series_rejects_thinking_parameter() -> None:
    spec = MODEL_SPECS[Model.KIMI_K2_0905_PREVIEW]
    with pytest.raises(ValueError, match="does not support the 'thinking' parameter"):
        spec.validate_params({"thinking": {"type": "enabled"}})


def test_k2_thinking_always_on_cannot_be_configured() -> None:
    spec = MODEL_SPECS[Model.KIMI_K2_THINKING]
    with pytest.raises(ValueError, match="always on"):
        spec.validate_params({"thinking": {"type": "disabled"}})


def test_k2_thinking_requires_max_tokens_floor() -> None:
    spec = MODEL_SPECS[Model.KIMI_K2_THINKING]
    with pytest.raises(ValueError, match="max_tokens >= 16000"):
        spec.validate_params({"max_tokens": 4_000})


def test_k2_thinking_requires_max_tokens_present() -> None:
    spec = MODEL_SPECS[Model.KIMI_K2_THINKING]
    with pytest.raises(ValueError, match="max_tokens >= 16000"):
        spec.validate_params({})


def test_k2_thinking_accepts_floor_value() -> None:
    spec = MODEL_SPECS[Model.KIMI_K2_THINKING]
    spec.validate_params({"max_tokens": 16_000})
    spec.validate_params({"max_tokens": 32_000})


def test_k2_thinking_turbo_shares_floor() -> None:
    spec = MODEL_SPECS[Model.KIMI_K2_THINKING_TURBO]
    with pytest.raises(ValueError, match="max_tokens >= 16000"):
        spec.validate_params({"max_tokens": 100})


def test_non_thinking_models_have_no_max_tokens_floor() -> None:
    """Models without max_tokens_min set must accept any value (including absent)."""
    spec = MODEL_SPECS[Model.KIMI_K2_0905_PREVIEW]
    spec.validate_params({})
    spec.validate_params({"max_tokens": 1})


def test_k2_series_allows_temperature_and_penalty() -> None:
    spec = MODEL_SPECS[Model.KIMI_K2_0905_PREVIEW]
    spec.validate_params({"temperature": 0.4, "presence_penalty": 0.5, "n": 3})


def test_n_max_enforced_for_k2_series() -> None:
    spec = MODEL_SPECS[Model.KIMI_K2_0905_PREVIEW]
    with pytest.raises(ValueError, match="n must be"):
        spec.validate_params({"n": 6})


def test_temperature_zero_disallows_n_gt_1() -> None:
    spec = MODEL_SPECS[Model.MOONSHOT_V1_8K]
    spec.validate_params({"temperature": 0.0, "n": 1})  # ok
    with pytest.raises(ValueError, match="temperature is 0, n must be 1"):
        spec.validate_params({"temperature": 0.0, "n": 2})


def test_moonshot_default_temperature_zero_with_implicit_n_gt_1() -> None:
    # n>1 with no temperature kwarg defaults to the model's default (0.0) and trips the rule.
    spec = MODEL_SPECS[Model.MOONSHOT_V1_8K]
    with pytest.raises(ValueError, match="temperature is 0, n must be 1"):
        spec.validate_params({"n": 2})


# --- tool_choice + thinking ---------------------------------------------------


def test_k26_rejects_tool_choice_required_when_thinking_enabled() -> None:
    spec = MODEL_SPECS[Model.KIMI_K2_6]
    with pytest.raises(ThinkingIncompatibilityError, match="tool_choice='required'"):
        spec.validate_params(
            {"tool_choice": "required", "thinking": {"type": "enabled"}}
        )


def test_k26_allows_tool_choice_required_when_thinking_omitted() -> None:
    spec = MODEL_SPECS[Model.KIMI_K2_6]
    spec.validate_params({"tool_choice": "required"})


def test_k26_allows_tool_choice_required_when_thinking_disabled() -> None:
    """Explicit thinking={'type': 'disabled'} → tool_choice='required' is fine."""
    spec = MODEL_SPECS[Model.KIMI_K2_6]
    spec.validate_params(
        {"tool_choice": "required", "thinking": {"type": "disabled"}}
    )


def test_k2_thinking_always_rejects_tool_choice_required() -> None:
    """Always-on thinking models reject 'required' unconditionally."""
    spec = MODEL_SPECS[Model.KIMI_K2_THINKING]
    with pytest.raises(ThinkingIncompatibilityError, match="always-on"):
        spec.validate_params({"tool_choice": "required", "max_tokens": 16_000})


def test_k2_thinking_turbo_shares_tool_choice_rule() -> None:
    spec = MODEL_SPECS[Model.KIMI_K2_THINKING_TURBO]
    with pytest.raises(ThinkingIncompatibilityError):
        spec.validate_params({"tool_choice": "required", "max_tokens": 16_000})


def test_non_thinking_model_allows_tool_choice_required() -> None:
    spec = MODEL_SPECS[Model.KIMI_K2_0905_PREVIEW]
    spec.validate_params({"tool_choice": "required"})


def test_tool_choice_auto_never_triggers_thinking_check() -> None:
    """Only the literal 'required' is rejected; 'auto' is fine everywhere."""
    spec = MODEL_SPECS[Model.KIMI_K2_THINKING]
    spec.validate_params({"tool_choice": "auto", "max_tokens": 16_000})


def test_thinking_incompatibility_is_a_value_error() -> None:
    """Existing `except ValueError` blocks still catch the new subclass."""
    spec = MODEL_SPECS[Model.KIMI_K2_6]
    with pytest.raises(ValueError):
        spec.validate_params(
            {"tool_choice": "required", "thinking": {"type": "enabled"}}
        )


