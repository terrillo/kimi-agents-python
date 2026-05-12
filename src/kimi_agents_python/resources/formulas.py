from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..types import FormulaFiber, FormulaToolList

if TYPE_CHECKING:
    from ..client import AsyncKimiClient, KimiClient
    from ..formulas import AsyncFormulaTool, FormulaTool


def _serialize_args(arguments: str | dict[str, Any]) -> str:
    return arguments if isinstance(arguments, str) else json.dumps(arguments)


class Formulas:
    """Official Moonshot Formula API resource.

    A Formula is a server-hosted Python script (e.g. ``moonshot/web-search:latest``)
    exposed as one or more tool definitions plus a ``/fibers`` invoke endpoint.
    See ``.platform/guide/use-official-tools.md``.
    """

    def __init__(self, client: KimiClient) -> None:
        self._client = client

    def tools(self, uri: str) -> list[dict[str, Any]]:
        """Fetch the tool defs published by a Formula URI."""
        response = self._client._request("GET", f"/formulas/{uri}/tools")
        return FormulaToolList.model_validate(response.json()).tools

    def invoke(
        self, uri: str, *, name: str, arguments: str | dict[str, Any]
    ) -> FormulaFiber:
        """Invoke one tool on a Formula. Returns the raw Fiber response."""
        body = {"name": name, "arguments": _serialize_args(arguments)}
        response = self._client._request(
            "POST", f"/formulas/{uri}/fibers", json=body
        )
        return FormulaFiber.model_validate(response.json())

    def load(self, uri: str) -> list[FormulaTool]:
        """Return :class:`FormulaTool` wrappers ready to drop into a Session.

        Each wrapper carries the Formula URI; when the tool loop dispatches
        ``.invoke(args)``, the wrapper POSTs to ``/formulas/{uri}/fibers`` and
        returns the Fiber's ``output_str`` as the tool-call result.

        When mounting tools from multiple Formulas, ensure each tool's
        ``function.name`` is unique across the combined set — the chat API
        rejects duplicate names with HTTP 401.
        """
        from ..formulas import FormulaTool

        return [FormulaTool.from_def(self, uri, d) for d in self.tools(uri)]


class AsyncFormulas:
    """Async mirror of :class:`Formulas`."""

    def __init__(self, client: AsyncKimiClient) -> None:
        self._client = client

    async def tools(self, uri: str) -> list[dict[str, Any]]:
        response = await self._client._request("GET", f"/formulas/{uri}/tools")
        return FormulaToolList.model_validate(response.json()).tools

    async def invoke(
        self, uri: str, *, name: str, arguments: str | dict[str, Any]
    ) -> FormulaFiber:
        body = {"name": name, "arguments": _serialize_args(arguments)}
        response = await self._client._request(
            "POST", f"/formulas/{uri}/fibers", json=body
        )
        return FormulaFiber.model_validate(response.json())

    async def load(self, uri: str) -> list[AsyncFormulaTool]:
        from ..formulas import AsyncFormulaTool

        defs = await self.tools(uri)
        return [AsyncFormulaTool.from_def(self, uri, d) for d in defs]
