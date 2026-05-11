from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._enums import BatchEndpoint
from ..types import Batch, BatchList

if TYPE_CHECKING:
    from ..client import AsyncKimiClient, KimiClient


class Batches:
    def __init__(self, client: KimiClient) -> None:
        self._client = client

    def create(
        self,
        *,
        input_file_id: str,
        endpoint: BatchEndpoint | str = BatchEndpoint.CHAT_COMPLETIONS,
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
    ) -> Batch:
        body: dict[str, Any] = {
            "input_file_id": input_file_id,
            "endpoint": str(endpoint),
            "completion_window": completion_window,
        }
        if metadata is not None:
            body["metadata"] = metadata
        response = self._client._request("POST", "/batches", json=body)
        return Batch.model_validate(response.json())

    def retrieve(self, batch_id: str) -> Batch:
        response = self._client._request("GET", f"/batches/{batch_id}")
        return Batch.model_validate(response.json())

    def list(
        self, *, after: str | None = None, limit: int | None = None
    ) -> BatchList:
        params: dict[str, Any] = {}
        if after is not None:
            params["after"] = after
        if limit is not None:
            params["limit"] = limit
        response = self._client._request(
            "GET", "/batches", params=params or None
        )
        return BatchList.model_validate(response.json())

    def cancel(self, batch_id: str) -> Batch:
        response = self._client._request("POST", f"/batches/{batch_id}/cancel")
        return Batch.model_validate(response.json())


class AsyncBatches:
    def __init__(self, client: AsyncKimiClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        input_file_id: str,
        endpoint: BatchEndpoint | str = BatchEndpoint.CHAT_COMPLETIONS,
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
    ) -> Batch:
        body: dict[str, Any] = {
            "input_file_id": input_file_id,
            "endpoint": str(endpoint),
            "completion_window": completion_window,
        }
        if metadata is not None:
            body["metadata"] = metadata
        response = await self._client._request("POST", "/batches", json=body)
        return Batch.model_validate(response.json())

    async def retrieve(self, batch_id: str) -> Batch:
        response = await self._client._request("GET", f"/batches/{batch_id}")
        return Batch.model_validate(response.json())

    async def list(
        self, *, after: str | None = None, limit: int | None = None
    ) -> BatchList:
        params: dict[str, Any] = {}
        if after is not None:
            params["after"] = after
        if limit is not None:
            params["limit"] = limit
        response = await self._client._request(
            "GET", "/batches", params=params or None
        )
        return BatchList.model_validate(response.json())

    async def cancel(self, batch_id: str) -> Batch:
        response = await self._client._request(
            "POST", f"/batches/{batch_id}/cancel"
        )
        return Batch.model_validate(response.json())
