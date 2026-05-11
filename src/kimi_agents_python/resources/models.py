from __future__ import annotations

from typing import TYPE_CHECKING

from ..types import ModelInfo, ModelList

if TYPE_CHECKING:
    from ..client import AsyncKimiClient, KimiClient


class Models:
    def __init__(self, client: KimiClient) -> None:
        self._client = client

    def list(self) -> list[ModelInfo]:
        response = self._client._request("GET", "/models")
        return ModelList.model_validate(response.json()).data


class AsyncModels:
    def __init__(self, client: AsyncKimiClient) -> None:
        self._client = client

    async def list(self) -> list[ModelInfo]:
        response = await self._client._request("GET", "/models")
        return ModelList.model_validate(response.json()).data
