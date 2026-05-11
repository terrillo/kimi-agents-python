from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ._enums import (
    FinishReason,
    Model,
    ResponseFormatType,
    Role,
    ThinkingMode,
    ToolChoice,
)


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True, use_enum_values=True)


# --- Content parts (multimodal) -------------------------------------------------


class TextPart(_Base):
    type: Literal["text"] = "text"
    text: str


class ImageUrl(_Base):
    url: str


class ImageUrlPart(_Base):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl | str


class VideoUrl(_Base):
    url: str


class VideoUrlPart(_Base):
    type: Literal["video_url"] = "video_url"
    video_url: VideoUrl | str


ContentPart = Annotated[
    TextPart | ImageUrlPart | VideoUrlPart,
    Field(discriminator="type"),
]


# --- Tool calls -----------------------------------------------------------------


class FunctionCall(_Base):
    name: str
    arguments: str  # JSON-encoded string per the API


class ToolCall(_Base):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class FunctionDef(_Base):
    name: str
    description: str | None = None
    parameters: dict[str, Any]
    strict: bool | None = None


class ToolDef(_Base):
    type: Literal["function"] = "function"
    function: FunctionDef


# --- Messages -------------------------------------------------------------------


class Message(_Base):
    role: Role
    content: str | list[ContentPart] | None = None
    name: str | None = None
    partial: bool | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    reasoning_content: str | None = None


# --- Request-side feature configs -----------------------------------------------


class ThinkingConfig(_Base):
    type: ThinkingMode = ThinkingMode.ENABLED
    keep: Literal["all"] | None = None


class JsonSchemaSpec(_Base):
    name: str
    strict: bool = True
    schema_: dict[str, Any] = Field(alias="schema")


class ResponseFormat(_Base):
    type: ResponseFormatType
    json_schema: JsonSchemaSpec | None = None


class StreamOptions(_Base):
    include_usage: bool | None = None


# --- Chat completion request ----------------------------------------------------


class ChatCompletionRequest(_Base):
    model: Model | str
    messages: list[Message]
    max_completion_tokens: int | None = Field(default=None, alias="max_tokens")
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stream: bool | None = None
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    response_format: ResponseFormat | None = None
    tools: list[ToolDef] | None = None
    tool_choice: ToolChoice | None = None
    prompt_cache_key: str | None = None
    safety_identifier: str | None = None
    thinking: ThinkingConfig | None = None


# --- Chat completion response ---------------------------------------------------


class Usage(_Base):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int | None = None


class AssistantMessage(_Base):
    role: Role = Role.ASSISTANT
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(_Base):
    index: int
    message: AssistantMessage
    finish_reason: FinishReason | None = None


class ChatCompletion(_Base):
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None


# --- Streaming chunks -----------------------------------------------------------


class ChoiceDelta(_Base):
    role: Role | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class StreamChoice(_Base):
    index: int
    delta: ChoiceDelta
    finish_reason: FinishReason | None = None


class ChatCompletionChunk(_Base):
    id: str
    object: str
    created: int
    model: str
    choices: list[StreamChoice]
    usage: Usage | None = None


# --- Auxiliary endpoints --------------------------------------------------------


class ModelInfo(_Base):
    id: str
    object: str = "model"
    created: int | None = None
    owned_by: str | None = None
    context_length: int | None = None
    supports_image_in: bool | None = None
    supports_video_in: bool | None = None
    supports_reasoning: bool | None = None


class ModelList(_Base):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class BalanceData(_Base):
    available_balance: float
    voucher_balance: float
    cash_balance: float


class BalanceInfo(_Base):
    code: int | None = None
    status: bool | None = None
    scode: str | None = None
    data: BalanceData


class TokenEstimateData(_Base):
    total_tokens: int


class TokenEstimate(_Base):
    data: TokenEstimateData
