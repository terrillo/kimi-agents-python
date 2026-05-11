from __future__ import annotations

import pytest

from kimi_agents_python import (
    ContentFilterError,
    EngineOverloadedError,
    ExceededCurrentQuotaError,
    IncorrectAPIKeyError,
    InvalidAuthenticationError,
    InvalidRequestError,
    KimiAPIError,
    KimiAuthenticationError,
    KimiBadRequestError,
    KimiNotFoundError,
    KimiPermissionError,
    KimiRateLimitError,
    KimiServerError,
    PermissionDeniedError,
    RateLimitReachedError,
    ResourceNotFoundError,
    ServerErrorResponse,
    UnexpectedOutputError,
)
from kimi_agents_python.exceptions import exception_for


@pytest.mark.parametrize(
    "status, expected",
    [
        (400, KimiBadRequestError),
        (401, KimiAuthenticationError),
        (403, KimiPermissionError),
        (404, KimiNotFoundError),
        (429, KimiRateLimitError),
        (500, KimiServerError),
        (502, KimiServerError),
        (503, KimiServerError),
        (418, KimiAPIError),  # unmapped 4xx falls back to base
    ],
)
def test_exception_for_status_only(status: int, expected: type[KimiAPIError]) -> None:
    assert exception_for(status) is expected


@pytest.mark.parametrize(
    "status, error_type, expected, parent",
    [
        (400, "content_filter", ContentFilterError, KimiBadRequestError),
        (400, "invalid_request_error", InvalidRequestError, KimiBadRequestError),
        (401, "invalid_authentication_error", InvalidAuthenticationError, KimiAuthenticationError),
        (401, "incorrect_api_key_error", IncorrectAPIKeyError, KimiAuthenticationError),
        (403, "permission_denied_error", PermissionDeniedError, KimiPermissionError),
        (404, "resource_not_found_error", ResourceNotFoundError, KimiNotFoundError),
        (429, "engine_overloaded_error", EngineOverloadedError, KimiRateLimitError),
        (429, "exceeded_current_quota_error", ExceededCurrentQuotaError, KimiRateLimitError),
        (429, "rate_limit_reached_error", RateLimitReachedError, KimiRateLimitError),
        (500, "server_error", ServerErrorResponse, KimiServerError),
        (500, "unexpected_output", UnexpectedOutputError, KimiServerError),
    ],
)
def test_exception_for_typed_error(
    status: int,
    error_type: str,
    expected: type[KimiAPIError],
    parent: type[KimiAPIError],
) -> None:
    cls = exception_for(status, error_type)
    assert cls is expected
    assert issubclass(cls, parent)


def test_exception_for_unknown_type_falls_back_to_status() -> None:
    assert exception_for(429, "totally_new_error_type") is KimiRateLimitError
    assert exception_for(404, None) is KimiNotFoundError


