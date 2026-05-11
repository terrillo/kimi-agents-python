from __future__ import annotations

from enum import StrEnum


class Model(StrEnum):
    KIMI_K2_6 = "kimi-k2.6"
    KIMI_K2_5 = "kimi-k2.5"
    KIMI_K2_0905_PREVIEW = "kimi-k2-0905-preview"
    KIMI_K2_0711_PREVIEW = "kimi-k2-0711-preview"
    KIMI_K2_TURBO_PREVIEW = "kimi-k2-turbo-preview"
    KIMI_K2_THINKING = "kimi-k2-thinking"
    KIMI_K2_THINKING_TURBO = "kimi-k2-thinking-turbo"
    MOONSHOT_V1_8K = "moonshot-v1-8k"
    MOONSHOT_V1_32K = "moonshot-v1-32k"
    MOONSHOT_V1_128K = "moonshot-v1-128k"
    MOONSHOT_V1_AUTO = "moonshot-v1-auto"
    MOONSHOT_V1_8K_VISION_PREVIEW = "moonshot-v1-8k-vision-preview"
    MOONSHOT_V1_32K_VISION_PREVIEW = "moonshot-v1-32k-vision-preview"
    MOONSHOT_V1_128K_VISION_PREVIEW = "moonshot-v1-128k-vision-preview"


AVAILABLE_MODELS: tuple[Model, ...] = tuple(Model)


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"


class ThinkingMode(StrEnum):
    ENABLED = "enabled"
    DISABLED = "disabled"


class ResponseFormatType(StrEnum):
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class ToolChoice(StrEnum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


class FilePurpose(StrEnum):
    FILE_EXTRACT = "file-extract"
    IMAGE = "image"
    VIDEO = "video"
    BATCH = "batch"


class ThinkingSupport(StrEnum):
    """How a model exposes the ``thinking`` capability to callers."""

    NONE = "none"
    CONFIGURABLE = "configurable"
    ALWAYS_ON = "always_on"


class BatchStatus(StrEnum):
    VALIDATING = "validating"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class BatchEndpoint(StrEnum):
    CHAT_COMPLETIONS = "/v1/chat/completions"
