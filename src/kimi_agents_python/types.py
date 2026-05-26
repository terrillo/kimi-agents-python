from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ._enums import (
    BatchStatus,
    FilePurpose,
    FinishReason,
    Model,
    ResponseFormatType,
    Role,
    ThinkingMode,
    ToolChoice,
)


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True, use_enum_values=True)


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


class BuiltinFunctionRef(_Base):
    name: str


class BuiltinToolDef(_Base):
    """Wire shape for a Kimi `builtin_function` tool (e.g. `$web_search`).

    Carries only ``type`` and ``function.name`` — the server owns the
    implementation, so no parameter schema is sent.
    """

    type: Literal["builtin_function"] = "builtin_function"
    function: BuiltinFunctionRef


class Message(_Base):
    """A single chat message in the conversation history.

    Multi-turn footgun for thinking-enabled requests: when an assistant turn
    comes back with ``reasoning_content`` set, you **must** include that field
    verbatim when echoing the assistant message back in the next request's
    ``messages`` list. Dropping it makes Moonshot return an opaque HTTP 400.

    This applies to ``kimi-k2.6`` / ``kimi-k2.5`` when called with
    ``thinking={"type": "enabled", "keep": "all"}``.

    :func:`run_tools` / :func:`arun_tools` already handle this for you;
    hand-built message lists need to copy ``reasoning_content`` along with
    ``role``, ``content``, and ``tool_calls``.
    """

    role: Role
    content: str | list[ContentPart] | None = None
    name: str | None = None
    partial: bool | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    reasoning_content: str | None = None


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
    tools: (
        list[Annotated[ToolDef | BuiltinToolDef, Field(discriminator="type")]] | None
    ) = None
    tool_choice: ToolChoice | None = None
    prompt_cache_key: str | None = None
    safety_identifier: str | None = None
    thinking: ThinkingConfig | None = None


class PromptTokensDetails(_Base):
    cached_tokens: int | None = None


class Usage(_Base):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int | None = None
    prompt_tokens_details: PromptTokensDetails | None = None


def _cached_tokens_from(usage: Usage | None) -> int:
    """Cached tokens from a Usage, preferring the nested wire location.

    Returns 0 when ``usage`` is None or carries no cached-token info. Matches
    the precedence used by :class:`CacheStats.record`.
    """
    if usage is None:
        return 0
    nested = (
        usage.prompt_tokens_details.cached_tokens
        if usage.prompt_tokens_details is not None
        else None
    )
    return nested if nested is not None else (usage.cached_tokens or 0)


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

    @property
    def cached_tokens(self) -> int:
        """Cached prompt tokens for this response, or 0 when unreported.

        Reads ``usage.prompt_tokens_details.cached_tokens`` and falls back to
        the legacy top-level ``usage.cached_tokens`` for older payloads.
        """
        return _cached_tokens_from(self.usage)


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

    @property
    def cached_tokens(self) -> int:
        """Cached prompt tokens carried by this chunk, or 0 when unreported.

        Usage is normally only present on the final chunk (when the caller
        passed ``stream_options={"include_usage": True}``).
        """
        return _cached_tokens_from(self.usage)


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


class FiberContext(_Base):
    input: str | None = None
    output: str | None = None
    encrypted_output: str | None = None
    error: str | None = None


class FormulaFiber(_Base):
    """Result of `POST /formulas/{uri}/fibers` — one Formula tool invocation."""

    id: str
    object: Literal["fiber"] = "fiber"
    created_at: int | None = None
    lambda_id: str | None = None
    status: str
    context: FiberContext = FiberContext()
    formula: str | None = None
    organization_id: str | None = None
    project_id: str | None = None
    error: str | None = None

    @property
    def output_str(self) -> str:
        """The value to put in the ``role=tool`` message back to the model.

        Prefers ``context.output`` (plain Formulas), falls back to
        ``context.encrypted_output`` (protected Formulas like ``web-search``).
        On non-``succeeded`` status, returns a stringified error so the model
        can react instead of the caller seeing a None.
        """
        if self.status != "succeeded":
            err = self.error or self.context.error or self.context.output
            return f"Error: {err or 'fiber failed'}"
        if self.context.output is not None:
            return self.context.output
        if self.context.encrypted_output is not None:
            return self.context.encrypted_output
        return ""


class FormulaToolList(_Base):
    """Result of `GET /formulas/{uri}/tools` — raw tool defs from a formula."""

    object: Literal["list"] = "list"
    tools: list[dict[str, Any]] = []


class TokenEstimateData(_Base):
    total_tokens: int


class TokenEstimate(_Base):
    data: TokenEstimateData


class FileObject(_Base):
    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: FilePurpose
    status: str
    status_details: str | None = None


class FileList(_Base):
    object: Literal["list"] = "list"
    data: list[FileObject]


class FileDeleted(_Base):
    id: str
    object: Literal["file"] = "file"
    deleted: bool


class BatchRequestCounts(_Base):
    completed: int = 0
    failed: int = 0
    total: int = 0


class Batch(_Base):
    id: str
    object: Literal["batch"] = "batch"
    endpoint: str
    input_file_id: str
    completion_window: str
    status: BatchStatus
    output_file_id: str | None = None
    error_file_id: str | None = None
    created_at: int
    in_progress_at: int | None = None
    expires_at: int | None = None
    finalizing_at: int | None = None
    completed_at: int | None = None
    failed_at: int | None = None
    cancelling_at: int | None = None
    cancelled_at: int | None = None
    request_counts: BatchRequestCounts | None = None
    metadata: dict[str, str] | None = None


class BatchList(_Base):
    object: Literal["list"] = "list"
    data: list[Batch]
    has_more: bool = False


T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class PrefilledCompletion:
    """Result of :meth:`Chat.prefill` — the raw completion plus the prefilled text.

    ``text`` is the prefill string concatenated with the model's continuation,
    so callers can use it directly without remembering to splice.
    """

    completion: ChatCompletion
    text: str


@dataclass(slots=True, frozen=True)
class ParsedChatCompletion(Generic[T]):
    """Result of :meth:`Chat.parse` — the raw completion plus a typed value."""

    completion: ChatCompletion
    parsed: T
