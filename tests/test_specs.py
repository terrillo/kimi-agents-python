from __future__ import annotations

import pytest

from kimi_agents_python import MODEL_SPECS, Model, ModelSpec, get_model_spec


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


def test_context_lengths() -> None:
    assert MODEL_SPECS[Model.MOONSHOT_V1_8K].context_length == 8_192
    assert MODEL_SPECS[Model.MOONSHOT_V1_32K].context_length == 32_768
    assert MODEL_SPECS[Model.MOONSHOT_V1_128K].context_length == 131_072
    assert MODEL_SPECS[Model.KIMI_K2_6].context_length == 262_144
    assert MODEL_SPECS[Model.KIMI_K2_0711_PREVIEW].context_length == 131_072


def test_vision_and_video_capabilities() -> None:
    assert MODEL_SPECS[Model.KIMI_K2_6].supports_vision is True
    assert MODEL_SPECS[Model.KIMI_K2_6].supports_video is True
    assert MODEL_SPECS[Model.MOONSHOT_V1_8K_VISION_PREVIEW].supports_vision is True
    assert MODEL_SPECS[Model.MOONSHOT_V1_8K_VISION_PREVIEW].supports_video is False
    assert MODEL_SPECS[Model.MOONSHOT_V1_8K].supports_vision is False


def test_families() -> None:
    assert MODEL_SPECS[Model.KIMI_K2_6].family == "kimi-k2.6"
    assert MODEL_SPECS[Model.KIMI_K2_5].family == "kimi-k2.5"
    assert MODEL_SPECS[Model.KIMI_K2_0905_PREVIEW].family == "kimi-k2"
    assert MODEL_SPECS[Model.KIMI_K2_THINKING].family == "kimi-k2-thinking"
    assert MODEL_SPECS[Model.MOONSHOT_V1_8K].family == "moonshot-v1"
    assert MODEL_SPECS[Model.MOONSHOT_V1_8K_VISION_PREVIEW].family == "moonshot-v1-vision"
