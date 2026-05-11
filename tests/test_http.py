from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from kimi_agents_python import (
    KimiAPIError,
    KimiAuthenticationError,
    KimiBadRequestError,
    KimiError,
    KimiRateLimitError,
    KimiServerError,
)
from kimi_agents_python import _http as http_mod
from kimi_agents_python._http import (
    bearer_headers,
    parse_sse_line,
    raise_for_status,
    resolve_api_key,
)


# --- parse_sse_line ------------------------------------------------------------


def test_parse_sse_line_data() -> None:
    assert parse_sse_line('data: {"a": 1}') == {"a": 1}


def test_parse_sse_line_empty() -> None:
    assert parse_sse_line("") is None


def test_parse_sse_line_done() -> None:
    assert parse_sse_line("data: [DONE]") is None


def test_parse_sse_line_non_data_prefix() -> None:
    assert parse_sse_line("event: ping") is None


def test_parse_sse_line_invalid_json_raises() -> None:
    with pytest.raises(KimiAPIError) as exc:
        parse_sse_line("data: {not json")
    assert exc.value.error_type == "invalid_stream_chunk"


# --- raise_for_status ----------------------------------------------------------


def _resp(status: int, json_body: dict | None = None, text: str | None = None) -> httpx.Response:
    if json_body is not None:
        return httpx.Response(status, json=json_body)
    return httpx.Response(status, content=(text or "").encode())


def test_raise_for_status_ok_passes() -> None:
    raise_for_status(_resp(200, {"ok": True}))


def test_raise_for_status_401_auth() -> None:
    r = _resp(
        401,
        {"error": {"type": "invalid_authentication_error", "message": "bad key"}},
    )
    with pytest.raises(KimiAuthenticationError) as exc:
        raise_for_status(r)
    assert exc.value.status_code == 401
    assert exc.value.error_type == "invalid_authentication_error"
    assert exc.value.message == "bad key"


def test_raise_for_status_400_bad_request() -> None:
    with pytest.raises(KimiBadRequestError):
        raise_for_status(_resp(400, {"error": {"type": "x", "message": "m"}}))


def test_raise_for_status_429_rate_limit() -> None:
    with pytest.raises(KimiRateLimitError):
        raise_for_status(_resp(429, {"error": {"type": "rl", "message": "slow"}}))


def test_raise_for_status_500_server() -> None:
    with pytest.raises(KimiServerError):
        raise_for_status(_resp(500, text="kaboom"))


def test_raise_for_status_falls_back_to_status_message() -> None:
    with pytest.raises(KimiAPIError) as exc:
        raise_for_status(_resp(418, text=""))
    assert "418" in exc.value.message


# --- bearer_headers ------------------------------------------------------------


def test_bearer_headers() -> None:
    assert bearer_headers("sk-abc") == {"Authorization": "Bearer sk-abc"}


# --- resolve_api_key -----------------------------------------------------------


def test_resolve_api_key_explicit_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "env-key")
    assert resolve_api_key("explicit-key") == "explicit-key"


def test_resolve_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "env-key")
    assert resolve_api_key(None) == "env-key"


def test_resolve_api_key_kimi_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIMI_API_KEY", "alt-key")
    assert resolve_api_key(None) == "alt-key"


def test_resolve_api_key_missing_raises() -> None:
    with pytest.raises(KimiError) as exc:
        resolve_api_key(None)
    assert ".env" in str(exc.value)


def test_resolve_api_key_loads_from_dotenv_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    http_mod._ensure_dotenv_loaded.cache_clear()
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("MOONSHOT_API_KEY=from-dotenv\n")
    assert resolve_api_key(None) == "from-dotenv"
