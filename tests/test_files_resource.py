from __future__ import annotations

from pathlib import Path

import httpx

from kimi_agents_python import FileDeleted, FileObject, FilePurpose
from tests.conftest import make_async_client, make_sync_client


def _file_body(file_id: str = "file-123") -> dict:
    return {
        "id": file_id,
        "object": "file",
        "bytes": 7,
        "created_at": 1700000000,
        "filename": "hello.txt",
        "purpose": "file-extract",
        "status": "ready",
    }


def test_files_create_sends_multipart(tmp_path: Path) -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/files"
        ctype = request.headers["content-type"]
        assert ctype.startswith("multipart/form-data")
        captured["body"] = request.content
        captured["ctype"] = ctype
        return httpx.Response(200, json=_file_body())

    p = tmp_path / "hello.txt"
    p.write_text("hello!")

    with make_sync_client(handler) as client:
        result = client.files.create(file=p, purpose=FilePurpose.FILE_EXTRACT)

    assert isinstance(result, FileObject)
    assert result.id == "file-123"
    assert b'name="purpose"\r\n\r\nfile-extract' in captured["body"]
    assert b"hello!" in captured["body"]
    assert b'filename="hello.txt"' in captured["body"]


def test_files_create_accepts_inline_bytes() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, json=_file_body())

    with make_sync_client(handler) as client:
        result = client.files.create(
            file=("inline.jsonl", b'{"x":1}\n'), purpose=FilePurpose.BATCH
        )
    assert result.id == "file-123"
    assert b'name="purpose"\r\n\r\nbatch' in captured["body"]
    assert b'filename="inline.jsonl"' in captured["body"]


def test_files_list_returns_data_array() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/files"
        return httpx.Response(
            200, json={"object": "list", "data": [_file_body("f1"), _file_body("f2")]}
        )

    with make_sync_client(handler) as client:
        files = client.files.list()
    assert [f.id for f in files] == ["f1", "f2"]


def test_files_retrieve_by_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/files/file-abc"
        return httpx.Response(200, json=_file_body("file-abc"))

    with make_sync_client(handler) as client:
        f = client.files.retrieve("file-abc")
    assert f.id == "file-abc"


def test_files_delete() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/v1/files/file-abc"
        return httpx.Response(
            200, json={"id": "file-abc", "object": "file", "deleted": True}
        )

    with make_sync_client(handler) as client:
        result = client.files.delete("file-abc")
    assert isinstance(result, FileDeleted)
    assert result.deleted is True


def test_files_content_returns_text_and_bytes() -> None:
    payload = "extracted text body"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/files/file-abc/content"
        return httpx.Response(
            200, content=payload.encode(), headers={"content-type": "text/plain"}
        )

    with make_sync_client(handler) as client:
        body = client.files.content("file-abc")
    assert body.text == payload
    assert body.content == payload.encode()


async def test_async_files_create() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_file_body())

    async with make_async_client(handler) as client:
        result = await client.files.create(
            file=("hello.txt", b"hi"), purpose=FilePurpose.FILE_EXTRACT
        )
    assert result.id == "file-123"


async def test_async_files_list_and_delete() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(
                200, json={"id": "file-x", "object": "file", "deleted": True}
            )
        return httpx.Response(
            200, json={"object": "list", "data": [_file_body("file-x")]}
        )

    async with make_async_client(handler) as client:
        files = await client.files.list()
        deleted = await client.files.delete("file-x")
    assert files[0].id == "file-x"
    assert deleted.deleted is True
