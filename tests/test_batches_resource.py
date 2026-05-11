from __future__ import annotations

import json

import httpx

from kimi_agents_python import Batch, BatchEndpoint, BatchList, BatchStatus
from tests.conftest import make_async_client, make_sync_client


def _batch_body(batch_id: str = "batch-1", status: str = "validating") -> dict:
    return {
        "id": batch_id,
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-in",
        "completion_window": "24h",
        "status": status,
        "created_at": 1700000000,
        "request_counts": {"completed": 0, "failed": 0, "total": 100},
    }


def test_batches_create_minimal() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/batches"
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_batch_body())

    with make_sync_client(handler) as client:
        result = client.batches.create(input_file_id="file-in")

    assert isinstance(result, Batch)
    assert result.id == "batch-1"
    assert result.status is BatchStatus.VALIDATING or result.status == "validating"
    assert captured["body"] == {
        "input_file_id": "file-in",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    }


def test_batches_create_with_metadata_and_window() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_batch_body())

    with make_sync_client(handler) as client:
        client.batches.create(
            input_file_id="file-in",
            endpoint=BatchEndpoint.CHAT_COMPLETIONS,
            completion_window="12h",
            metadata={"job": "nightly", "owner": "team-x"},
        )

    assert captured["body"]["completion_window"] == "12h"
    assert captured["body"]["metadata"] == {"job": "nightly", "owner": "team-x"}


def test_batches_retrieve() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/batches/batch-9"
        return httpx.Response(200, json=_batch_body("batch-9", status="completed"))

    with make_sync_client(handler) as client:
        b = client.batches.retrieve("batch-9")
    assert b.id == "batch-9"
    assert str(b.status) == "completed"


def test_batches_list_pagination_params() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["query"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [_batch_body("a"), _batch_body("b")],
                "has_more": True,
            },
        )

    with make_sync_client(handler) as client:
        page = client.batches.list(after="cursor-x", limit=50)

    assert isinstance(page, BatchList)
    assert page.has_more is True
    assert [b.id for b in page.data] == ["a", "b"]
    assert captured["query"] == {"after": "cursor-x", "limit": "50"}


def test_batches_list_no_params_sends_no_query() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["query"] = str(request.url.params)
        return httpx.Response(
            200, json={"object": "list", "data": [], "has_more": False}
        )

    with make_sync_client(handler) as client:
        client.batches.list()

    assert captured["query"] == ""


def test_batches_cancel() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/batches/batch-9/cancel"
        return httpx.Response(200, json=_batch_body("batch-9", status="cancelling"))

    with make_sync_client(handler) as client:
        b = client.batches.cancel("batch-9")
    assert str(b.status) == "cancelling"


async def test_async_batches_create_and_retrieve() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json=_batch_body())
        return httpx.Response(200, json=_batch_body(status="completed"))

    async with make_async_client(handler) as client:
        created = await client.batches.create(input_file_id="file-in")
        fetched = await client.batches.retrieve(created.id)
    assert created.id == "batch-1"
    assert str(fetched.status) == "completed"
