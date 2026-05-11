from __future__ import annotations


class KimiError(Exception):
    pass


class KimiAPIError(KimiError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str | None = None,
        error_code: str | None = None,
        raw: object = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code
        self.raw = raw

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


_STATUS_TO_EXC: dict[int, type[KimiAPIError]] = {
    400: KimiBadRequestError,
    401: KimiAuthenticationError,
    403: KimiPermissionError,
    404: KimiNotFoundError,
    429: KimiRateLimitError,
}


def exception_for_status(status_code: int) -> type[KimiAPIError]:
    if status_code in _STATUS_TO_EXC:
        return _STATUS_TO_EXC[status_code]
    if status_code >= 500:
        return KimiServerError
    return KimiAPIError
