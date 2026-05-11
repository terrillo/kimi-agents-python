from __future__ import annotations

from kimi_agents_python import AVAILABLE_MODELS, Model


def test_all_expected_model_ids_present() -> None:
    """Freeze test: the public wire strings the package commits to supporting."""
    expected = {
        "kimi-k2.6",
        "kimi-k2.5",
        "kimi-k2-0905-preview",
        "kimi-k2-0711-preview",
        "kimi-k2-turbo-preview",
        "kimi-k2-thinking",
        "kimi-k2-thinking-turbo",
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshot-v1-128k",
        "moonshot-v1-auto",
        "moonshot-v1-8k-vision-preview",
        "moonshot-v1-32k-vision-preview",
        "moonshot-v1-128k-vision-preview",
    }
    assert {m.value for m in Model} == expected
    assert set(AVAILABLE_MODELS) == set(Model)
