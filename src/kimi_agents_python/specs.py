from __future__ import annotations

from dataclasses import dataclass

from ._enums import Model, ThinkingSupport


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Capabilities and parameter constraints for a single model.

    Mirrors the per-family rules at https://platform.kimi.ai/docs/api/models-overview
    and the model list at https://platform.kimi.ai/docs/models.
    """

    id: Model
    family: str
    context_length: int

    supports_vision: bool = False
    supports_video: bool = False
    supports_tools: bool = True
    supports_json_schema: bool = True
    supports_partial: bool = True

    thinking_support: ThinkingSupport = ThinkingSupport.NONE

    temperature_default: float = 0.6
    temperature_locked: bool = False
    top_p_default: float = 1.0
    top_p_locked: bool = False
    n_default: int = 1
    n_max: int = 5
    n_locked: bool = False
    penalty_locked: bool = False

    def validate_params(self, params: dict) -> None:
        """Raise ValueError if any kwarg in ``params`` violates this model's rules.

        ``params`` is the chat() kwargs (everything except ``model`` and ``messages``).
        Unknown / unrecognised keys are ignored — pydantic catches them downstream.
        """
        self._check_locked("temperature", params, self.temperature_locked, self.temperature_default)
        self._check_locked("top_p", params, self.top_p_locked, self.top_p_default)
        self._check_n(params)
        self._check_penalty(params)
        self._check_thinking(params)

    def _check_locked(self, key: str, params: dict, locked: bool, default: float) -> None:
        if locked and key in params and params[key] != default:
            raise ValueError(
                f"{self.id.value}: {key} is locked at {default}; got {params[key]!r}."
            )

    def _check_n(self, params: dict) -> None:
        if "n" not in params:
            return
        n = params["n"]
        if self.n_locked and n != self.n_default:
            raise ValueError(
                f"{self.id.value}: n is locked at {self.n_default}; got {n!r}."
            )
        if n > self.n_max:
            raise ValueError(
                f"{self.id.value}: n must be ≤ {self.n_max}; got {n}."
            )
        temp = params.get("temperature", self.temperature_default)
        if n > 1 and temp <= 1e-6:
            raise ValueError(
                f"{self.id.value}: when temperature is 0, n must be 1."
            )

    def _check_penalty(self, params: dict) -> None:
        if not self.penalty_locked:
            return
        for key in ("presence_penalty", "frequency_penalty"):
            if key in params and params[key] != 0:
                raise ValueError(
                    f"{self.id.value}: {key} is locked at 0; got {params[key]!r}."
                )

    def _check_thinking(self, params: dict) -> None:
        if "thinking" not in params:
            return
        if self.thinking_support is ThinkingSupport.NONE:
            raise ValueError(
                f"{self.id.value}: does not support the 'thinking' parameter."
            )
        if self.thinking_support is ThinkingSupport.ALWAYS_ON:
            raise ValueError(
                f"{self.id.value}: 'thinking' is always on for this model and "
                f"cannot be configured."
            )


_K26_LOCKED = dict(
    family="kimi-k2.6",
    context_length=262_144,
    supports_vision=True,
    supports_video=True,
    thinking_support=ThinkingSupport.CONFIGURABLE,
    temperature_default=1.0,
    temperature_locked=True,
    top_p_default=0.95,
    top_p_locked=True,
    n_default=1,
    n_max=1,
    n_locked=True,
    penalty_locked=True,
)


MODEL_SPECS: dict[Model, ModelSpec] = {
    Model.KIMI_K2_6: ModelSpec(id=Model.KIMI_K2_6, **_K26_LOCKED),
    Model.KIMI_K2_5: ModelSpec(
        id=Model.KIMI_K2_5, **{**_K26_LOCKED, "family": "kimi-k2.5"}
    ),
    Model.KIMI_K2_0905_PREVIEW: ModelSpec(
        id=Model.KIMI_K2_0905_PREVIEW,
        family="kimi-k2",
        context_length=262_144,
        temperature_default=0.6,
    ),
    Model.KIMI_K2_0711_PREVIEW: ModelSpec(
        id=Model.KIMI_K2_0711_PREVIEW,
        family="kimi-k2",
        context_length=131_072,
        temperature_default=0.6,
    ),
    Model.KIMI_K2_TURBO_PREVIEW: ModelSpec(
        id=Model.KIMI_K2_TURBO_PREVIEW,
        family="kimi-k2",
        context_length=262_144,
        temperature_default=0.6,
    ),
    Model.KIMI_K2_THINKING: ModelSpec(
        id=Model.KIMI_K2_THINKING,
        family="kimi-k2-thinking",
        context_length=262_144,
        thinking_support=ThinkingSupport.ALWAYS_ON,
        temperature_default=1.0,
    ),
    Model.KIMI_K2_THINKING_TURBO: ModelSpec(
        id=Model.KIMI_K2_THINKING_TURBO,
        family="kimi-k2-thinking",
        context_length=262_144,
        thinking_support=ThinkingSupport.ALWAYS_ON,
        temperature_default=1.0,
    ),
    Model.MOONSHOT_V1_8K: ModelSpec(
        id=Model.MOONSHOT_V1_8K,
        family="moonshot-v1",
        context_length=8_192,
        temperature_default=0.0,
    ),
    Model.MOONSHOT_V1_32K: ModelSpec(
        id=Model.MOONSHOT_V1_32K,
        family="moonshot-v1",
        context_length=32_768,
        temperature_default=0.0,
    ),
    Model.MOONSHOT_V1_128K: ModelSpec(
        id=Model.MOONSHOT_V1_128K,
        family="moonshot-v1",
        context_length=131_072,
        temperature_default=0.0,
    ),
    Model.MOONSHOT_V1_AUTO: ModelSpec(
        id=Model.MOONSHOT_V1_AUTO,
        family="moonshot-v1",
        context_length=131_072,
        temperature_default=0.0,
    ),
    Model.MOONSHOT_V1_8K_VISION_PREVIEW: ModelSpec(
        id=Model.MOONSHOT_V1_8K_VISION_PREVIEW,
        family="moonshot-v1-vision",
        context_length=8_192,
        supports_vision=True,
        temperature_default=0.0,
    ),
    Model.MOONSHOT_V1_32K_VISION_PREVIEW: ModelSpec(
        id=Model.MOONSHOT_V1_32K_VISION_PREVIEW,
        family="moonshot-v1-vision",
        context_length=32_768,
        supports_vision=True,
        temperature_default=0.0,
    ),
    Model.MOONSHOT_V1_128K_VISION_PREVIEW: ModelSpec(
        id=Model.MOONSHOT_V1_128K_VISION_PREVIEW,
        family="moonshot-v1-vision",
        context_length=131_072,
        supports_vision=True,
        temperature_default=0.0,
    ),
}


def get_model_spec(model: Model | str) -> ModelSpec | None:
    """Return the spec for a model id, or ``None`` if the id is unknown.

    Unknown ids are intentionally tolerated so new models the server adds
    don't require a client release to be usable.
    """
    if isinstance(model, Model):
        return MODEL_SPECS.get(model)
    try:
        return MODEL_SPECS[Model(model)]
    except ValueError:
        return None
