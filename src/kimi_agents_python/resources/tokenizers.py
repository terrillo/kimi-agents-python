from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .._enums import Model
from ..types import Message, TokenEstimate

if TYPE_CHECKING:
    from ..client import AsyncKimiClient, KimiClient


_MessageInput = Message | dict[str, Any]


class Tokenizers:
    def __init__(self, client: KimiClient) -> None:
        self._client = client

    def estimate(
        self, *, model: Model | str, messages: Iterable[_MessageInput]
    ) -> TokenEstimate:
        body = {"model": str(model), "messages": list(messages)}
        response = self._client._request(
            "POST", "/tokenizers/estimate-token-count", json=body
        )
        return TokenEstimate.model_validate(response.json())


class AsyncTokenizers:
    def __init__(self, client: AsyncKimiClient) -> None:
        self._client = client

    async def estimate(
        self, *, model: Model | str, messages: Iterable[_MessageInput]
    ) -> TokenEstimate:
        body = {"model": str(model), "messages": list(messages)}
        response = await self._client._request(
            "POST", "/tokenizers/estimate-token-count", json=body
        )
        return TokenEstimate.model_validate(response.json())
