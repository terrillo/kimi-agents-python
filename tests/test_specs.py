from __future__ import annotations

import pytest

from kimi_agents_python import (
    MODEL_SPECS,
    Model,
    ModelSpec,
    ThinkingIncompatibilityError,
    get_model_spec,
)
from kimi_agents_python._enums import ThinkingSupport


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


def test_k26_real_spec_invariants() -> None:
    """Pin the shipped KIMI_K2_6 capabilities so a regression that flips them
    (e.g. accidentally setting max_tokens_min or ALWAYS_ON) is caught here, not
    at the wire. The synthetic-spec tests below build their own ModelSpec — they
    would not detect a corruption of the real production entry."""
    spec = MODEL_SPECS[Model.KIMI_K2_6]
    assert spec.thinking_support is ThinkingSupport.CONFIGURABLE
    assert spec.max_tokens_min is None
    assert spec.temperature_locked is True
    assert spec.n_locked is True


def test_n_max_enforced_when_unlocked() -> None:
    """Models with n_locked=False still cap n at n_max."""
    spec = ModelSpec(
        id=Model.KIMI_K2_6,
        family="synthetic",
        context_length=1,
        temperature_default=0.6,
        n_max=5,
    )
    spec.validate_params({"n": 5})  # at the cap → fine
    with pytest.raises(ValueError, match="n must be"):
        spec.validate_params({"n": 6})


def test_moonshot_rejects_thinking_parameter() -> None:
    spec = MODEL_SPECS[Model.MOONSHOT_V1_8K]
    with pytest.raises(ValueError, match="does not support the 'thinking' parameter"):
        spec.validate_params({"thinking": {"type": "enabled"}})


def test_always_on_thinking_cannot_be_configured() -> None:
    spec = ModelSpec(
        id=Model.KIMI_K2_6,
        family="synthetic",
        context_length=1,
        thinking_support=ThinkingSupport.ALWAYS_ON,
    )
    with pytest.raises(ValueError, match="always on"):
        spec.validate_params({"thinking": {"type": "disabled"}})


def test_max_tokens_floor_required() -> None:
    spec = ModelSpec(
        id=Model.KIMI_K2_6,
        family="synthetic",
        context_length=1,
        max_tokens_min=16_000,
    )
    with pytest.raises(ValueError, match="max_tokens >= 16000"):
        spec.validate_params({"max_tokens": 4_000})
    with pytest.raises(ValueError, match="max_tokens >= 16000"):
        spec.validate_params({})
    spec.validate_params({"max_tokens": 16_000})
    spec.validate_params({"max_tokens": 32_000})


def test_moonshot_has_no_max_tokens_floor() -> None:
    spec = MODEL_SPECS[Model.MOONSHOT_V1_8K]
    spec.validate_params({"temperature": 0.0})
    spec.validate_params({"temperature": 0.0, "max_tokens": 1})


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


def test_always_on_thinking_rejects_tool_choice_required() -> None:
    spec = ModelSpec(
        id=Model.KIMI_K2_6,
        family="synthetic",
        context_length=1,
        thinking_support=ThinkingSupport.ALWAYS_ON,
    )
    with pytest.raises(ThinkingIncompatibilityError, match="always-on"):
        spec.validate_params({"tool_choice": "required"})


def test_non_thinking_model_allows_tool_choice_required() -> None:
    spec = MODEL_SPECS[Model.MOONSHOT_V1_8K]
    spec.validate_params({"tool_choice": "required"})


def test_tool_choice_auto_never_triggers_thinking_check() -> None:
    """Only the literal 'required' is rejected; 'auto' is fine everywhere."""
    spec = ModelSpec(
        id=Model.KIMI_K2_6,
        family="synthetic",
        context_length=1,
        thinking_support=ThinkingSupport.ALWAYS_ON,
    )
    spec.validate_params({"tool_choice": "auto"})


def test_thinking_incompatibility_is_a_value_error() -> None:
    """Existing `except ValueError` blocks still catch the new subclass."""
    spec = MODEL_SPECS[Model.KIMI_K2_6]
    with pytest.raises(ValueError):
        spec.validate_params(
            {"tool_choice": "required", "thinking": {"type": "enabled"}}
        )


@pytest.mark.parametrize("model", list(Model))
def test_every_model_accepts_json_schema_response_format(model: Model) -> None:
    """Structured output (json_schema) is honoured by every known model.

    All current families set ``supports_json_schema=True``; this pins that so
    a regression that flips a family off is caught here, not at the wire.
    """
    spec = MODEL_SPECS[model]
    assert spec.supports_json_schema is True
    params: dict = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "S", "schema": {"type": "object"}},
        }
    }
    if spec.max_tokens_min is not None:
        params["max_tokens"] = spec.max_tokens_min
    spec.validate_params(params)  # must not raise


def test_json_object_response_format_is_always_allowed() -> None:
    """json_object mode needs no special capability — allowed even if a model
    were to drop json_schema support."""
    spec = ModelSpec(
        id=Model.KIMI_K2_6, family="x", context_length=1, supports_json_schema=False
    )
    spec.validate_params({"response_format": {"type": "json_object"}})


def test_json_schema_rejected_when_model_lacks_support() -> None:
    spec = ModelSpec(
        id=Model.KIMI_K2_6, family="x", context_length=1, supports_json_schema=False
    )
    with pytest.raises(ValueError, match="does not support response_format"):
        spec.validate_params(
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "S", "schema": {}},
                }
            }
        )
