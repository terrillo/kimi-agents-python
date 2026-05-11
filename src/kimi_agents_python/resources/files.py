from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from .._enums import FilePurpose
from ..files import validate_file
from ..types import FileDeleted, FileList, FileObject

if TYPE_CHECKING:
    from ..client import AsyncKimiClient, KimiClient


_FileInput = str | Path | tuple[str, bytes] | tuple[str, IO[bytes]]


class FileContent:
    """Body of GET /files/{file_id}/content as both text and bytes."""

    __slots__ = ("text", "content")

    def __init__(self, *, text: str, content: bytes) -> None:
        self.text = text
        self.content = content


def _multipart_payload(
    filename: str, payload: Any, purpose: FilePurpose | str
) -> tuple[dict[str, Any], dict[str, str]]:
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return {"file": (filename, payload, mime)}, {"purpose": str(purpose)}


def _validate_path_input(
    file: _FileInput, purpose: FilePurpose | str
) -> Path | None:
    """Return the path when ``file`` is a path-like input, else ``None``.

    Runs ``validate_file`` against ``purpose`` when both are known so format
    errors fail fast before opening the file handle.
    """
    if not isinstance(file, (str, Path)):
        return None
    path = Path(file)
    if isinstance(purpose, FilePurpose):
        validate_file(path, purpose)
    return path


class Files:
    def __init__(self, client: KimiClient) -> None:
        self._client = client

    def create(
        self, *, file: _FileInput, purpose: FilePurpose | str
    ) -> FileObject:
        path = _validate_path_input(file, purpose)
        if path is not None:
            with path.open("rb") as fh:
                files, data = _multipart_payload(path.name, fh, purpose)
                response = self._client._request(
                    "POST", "/files", files=files, data=data
                )
        else:
            filename, payload = file  # type: ignore[misc]
            files, data = _multipart_payload(filename, payload, purpose)
            response = self._client._request(
                "POST", "/files", files=files, data=data
            )
        return FileObject.model_validate(response.json())

    def list(self) -> list[FileObject]:
        response = self._client._request("GET", "/files")
        return FileList.model_validate(response.json()).data

    def retrieve(self, file_id: str) -> FileObject:
        response = self._client._request("GET", f"/files/{file_id}")
        return FileObject.model_validate(response.json())

    def delete(self, file_id: str) -> FileDeleted:
        response = self._client._request("DELETE", f"/files/{file_id}")
        return FileDeleted.model_validate(response.json())

    def content(self, file_id: str) -> FileContent:
        response = self._client._request("GET", f"/files/{file_id}/content")
        return FileContent(text=response.text, content=response.content)


class AsyncFiles:
    def __init__(self, client: AsyncKimiClient) -> None:
        self._client = client

    async def create(
        self, *, file: _FileInput, purpose: FilePurpose | str
    ) -> FileObject:
        path = _validate_path_input(file, purpose)
        if path is not None:
            with path.open("rb") as fh:
                files, data = _multipart_payload(path.name, fh, purpose)
                response = await self._client._request(
                    "POST", "/files", files=files, data=data
                )
        else:
            filename, payload = file  # type: ignore[misc]
            files, data = _multipart_payload(filename, payload, purpose)
            response = await self._client._request(
                "POST", "/files", files=files, data=data
            )
        return FileObject.model_validate(response.json())

    async def list(self) -> list[FileObject]:
        response = await self._client._request("GET", "/files")
        return FileList.model_validate(response.json()).data

    async def retrieve(self, file_id: str) -> FileObject:
        response = await self._client._request("GET", f"/files/{file_id}")
        return FileObject.model_validate(response.json())

    async def delete(self, file_id: str) -> FileDeleted:
        response = await self._client._request("DELETE", f"/files/{file_id}")
        return FileDeleted.model_validate(response.json())

    async def content(self, file_id: str) -> FileContent:
        response = await self._client._request(
            "GET", f"/files/{file_id}/content"
        )
        return FileContent(text=response.text, content=response.content)
