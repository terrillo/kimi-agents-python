from __future__ import annotations

import json
import os
from functools import cache
from typing import Any

import httpx
from dotenv import find_dotenv, load_dotenv

from .exceptions import KimiAPIError, KimiError, exception_for

DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_TIMEOUT = 60.0
_SSE_DONE = "[DONE]"


@cache
def _ensure_dotenv_loaded() -> None:
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path)


def resolve_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    _ensure_dotenv_loaded()
    key = os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY")
    if not key:
        raise KimiError(
            "No API key provided. Pass api_key=..., set MOONSHOT_API_KEY in the "
            "environment, or add it to a .env file."
        )
    return key


def bearer_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    body: Any
    try:
        body = response.json()
    except ValueError:
        body = response.text

    err_type: str | None = None
    err_code: str | None = None
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        err = body["error"]
        err_type = err.get("type")
        err_code = err.get("code")
        message = err.get("message") or f"HTTP {response.status_code}"
    elif isinstance(body, str) and body:
        message = body
    else:
        message = f"HTTP {response.status_code}"

    raise exception_for(response.status_code, err_type)(
        message,
        status_code=response.status_code,
        error_type=err_type,
        error_code=err_code,
        raw=body,
    )


def parse_sse_line(line: str) -> dict[str, Any] | None:
    """Return parsed JSON dict for a `data: {...}` line, None for terminator/other."""
    if not line or not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == _SSE_DONE:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        raise KimiAPIError(
            f"Failed to decode SSE chunk: {e}",
            status_code=0,
            error_type="invalid_stream_chunk",
            raw=payload,
        ) from e
