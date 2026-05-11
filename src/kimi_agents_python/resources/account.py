from __future__ import annotations

from typing import TYPE_CHECKING

from ..types import BalanceInfo

if TYPE_CHECKING:
    from ..client import AsyncKimiClient, KimiClient


class Account:
    def __init__(self, client: KimiClient) -> None:
        self._client = client

    def balance(self) -> BalanceInfo:
        response = self._client._request("GET", "/users/me/balance")
        return BalanceInfo.model_validate(response.json())


class AsyncAccount:
    def __init__(self, client: AsyncKimiClient) -> None:
        self._client = client

    async def balance(self) -> BalanceInfo:
        response = await self._client._request("GET", "/users/me/balance")
        return BalanceInfo.model_validate(response.json())
