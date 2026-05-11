from __future__ import annotations


class KimiError(Exception):
    pass


class KimiToolLoopError(KimiError):
    """run_tools / arun_tools terminated by a budget or safety guard.

    Raised directly when ``max_steps`` is exhausted. Subclasses are raised
    when an explicit :class:`LoopGuards` limit is hit:

    * :class:`TokenBudgetExceededError` — cumulative usage over ``max_tokens``
    * :class:`ReadOnlyStreakExceededError` — too many consecutive read-only calls
    * :class:`RepeatedToolCallError` — same tool+args invoked N times in a row
    """


class TokenBudgetExceededError(KimiToolLoopError):
    """Cumulative ``usage.total_tokens`` across the loop exceeded ``LoopGuards.max_tokens``."""


class ReadOnlyStreakExceededError(KimiToolLoopError):
    """Loop made N consecutive read-only tool calls without a mutating call between them."""


class RepeatedToolCallError(KimiToolLoopError):
    """Same (tool name, arguments) appeared N times in a row — likely a stuck loop."""


class KimiAPIError(KimiError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str | None = None,
        error_code: str | None = None,
        raw: object = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code
        self.raw = raw
        self.retry_after = retry_after

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(status_code={self.status_code}, "
            f"error_type={self.error_type!r}, message={self.message!r})"
        )


class KimiBadRequestError(KimiAPIError):
    pass


class KimiAuthenticationError(KimiAPIError):
    pass


class KimiPermissionError(KimiAPIError):
    pass


class KimiNotFoundError(KimiAPIError):
    pass


class KimiRateLimitError(KimiAPIError):
    pass


class KimiServerError(KimiAPIError):
    pass


# Typed subclasses keyed on the `error.type` string the API returns.
# See https://platform.kimi.ai/docs/api/errors for the full taxonomy.

class ContentFilterError(KimiBadRequestError):
    """error.type = content_filter"""


class InvalidRequestError(KimiBadRequestError):
    """error.type = invalid_request_error"""


class InvalidAuthenticationError(KimiAuthenticationError):
    """error.type = invalid_authentication_error"""


class IncorrectAPIKeyError(KimiAuthenticationError):
    """error.type = incorrect_api_key_error"""


class PermissionDeniedError(KimiPermissionError):
    """error.type = permission_denied_error"""


class ResourceNotFoundError(KimiNotFoundError):
    """error.type = resource_not_found_error"""


class EngineOverloadedError(KimiRateLimitError):
    """error.type = engine_overloaded_error"""


class ExceededCurrentQuotaError(KimiRateLimitError):
    """error.type = exceeded_current_quota_error"""


class RateLimitReachedError(KimiRateLimitError):
    """error.type = rate_limit_reached_error"""


class ServerErrorResponse(KimiServerError):
    """error.type = server_error"""


class UnexpectedOutputError(KimiServerError):
    """error.type = unexpected_output"""


_STATUS_TO_EXC: dict[int, type[KimiAPIError]] = {
    400: KimiBadRequestError,
    401: KimiAuthenticationError,
    403: KimiPermissionError,
    404: KimiNotFoundError,
    429: KimiRateLimitError,
}


_TYPE_TO_EXC: dict[str, type[KimiAPIError]] = {
    "content_filter": ContentFilterError,
    "invalid_request_error": InvalidRequestError,
    "invalid_authentication_error": InvalidAuthenticationError,
    "incorrect_api_key_error": IncorrectAPIKeyError,
    "permission_denied_error": PermissionDeniedError,
    "resource_not_found_error": ResourceNotFoundError,
    "engine_overloaded_error": EngineOverloadedError,
    "exceeded_current_quota_error": ExceededCurrentQuotaError,
    "rate_limit_reached_error": RateLimitReachedError,
    "server_error": ServerErrorResponse,
    "unexpected_output": UnexpectedOutputError,
}


def exception_for(
    status_code: int, error_type: str | None = None
) -> type[KimiAPIError]:
    """Pick the most specific exception class for an API error response.

    Prefers the documented ``error.type`` subclass; falls back to the
    HTTP-status mapping; finally returns ``KimiServerError`` for 5xx and
    ``KimiAPIError`` for anything else.
    """
    if error_type and (cls := _TYPE_TO_EXC.get(error_type)):
        return cls
    if (cls := _STATUS_TO_EXC.get(status_code)):
        return cls
    if status_code >= 500:
        return KimiServerError
    return KimiAPIError
