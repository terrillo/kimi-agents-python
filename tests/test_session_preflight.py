from __future__ import annotations

import json

import httpx

from kimi_agents_python import AsyncSession, Model, Session

from .conftest import make_async_client, make_sync_client


def _estimate_handler(captured: dict) -> httpx.Response:
    def _h(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"data": {"total_tokens": 123}})

    return _h  # type: ignore[return-value]


def test_estimated_tokens_includes_pending_user_turn() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"data": {"total_tokens": 123}})

    client = make_sync_client(handler)
    session = Session(client, model=Model.KIMI_K2_6, system="sys")
    n = session.estimated_tokens("how many tokens is this?")
    assert n == 123
    assert captured["body"]["model"] == "kimi-k2.6"
    assert captured["body"]["messages"][-1]["role"] == "user"
    assert (
        captured["body"]["messages"][-1]["content"] == "how many tokens is this?"
    )
    # Session history unchanged after preflight.
    assert all(m.role != "user" for m in session.history)


async def test_async_estimated_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"total_tokens": 7}})

    client = make_async_client(handler)
    session = AsyncSession(client, model=Model.KIMI_K2_6)
    assert await session.estimated_tokens("hi") == 7
