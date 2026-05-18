from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from kimi_agents_python import AsyncKimiClient, KimiClient, Model
from kimi_agents_python import _http as http_mod

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hermetic env for every test: no API keys, dotenv cache cleared, cwd is empty."""
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    http_mod._ensure_dotenv_loaded.cache_clear()
    monkeypatch.chdir(tmp_path)


def make_sync_client(
    handler: Handler, *, api_key: str = "test-key", retries: int = 0
) -> KimiClient:
    http_client = httpx.Client(
        base_url="https://api.moonshot.ai/v1",
        transport=httpx.MockTransport(handler),
    )
    return KimiClient(api_key=api_key, http_client=http_client, retries=retries)


def make_async_client(
    handler: Handler, *, api_key: str = "test-key", retries: int = 0
) -> AsyncKimiClient:
    http_client = httpx.AsyncClient(
        base_url="https://api.moonshot.ai/v1",
        transport=httpx.MockTransport(handler),
    )
    return AsyncKimiClient(api_key=api_key, http_client=http_client, retries=retries)


def sse_response(chunks: list[dict | str]) -> httpx.Response:
    """Build an httpx Response that streams SSE chunks. Strings are sent verbatim."""
    body_parts = []
    for c in chunks:
        if isinstance(c, str):
            body_parts.append(f"data: {c}\n\n")
        else:
            body_parts.append(f"data: {json.dumps(c)}\n\n")
    return httpx.Response(
        200,
        content="".join(body_parts).encode("utf-8"),
        headers={"content-type": "text/event-stream"},
    )


def completion_body(content: str = "hello", finish_reason: str = "stop") -> dict:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": Model.KIMI_K2_6.value,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
