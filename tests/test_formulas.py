from __future__ import annotations

import json

import httpx
import pytest

from kimi_agents_python import (
    AsyncFormulaTool,
    AsyncKimiClient,
    FormulaFiber,
    FormulaTool,
)

from .conftest import make_async_client, make_sync_client


_TOOL_LIST = {
    "object": "list",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ],
}


def test_formulas_tools_parses_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/formulas/moonshot/web-search:latest/tools")
        return httpx.Response(200, json=_TOOL_LIST)

    client = make_sync_client(handler)
    tools = client.formulas.tools("moonshot/web-search:latest")
    assert tools[0]["function"]["name"] == "web_search"


def test_formulas_invoke_serializes_dict_args() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json={
                "id": "fiber-1",
                "object": "fiber",
                "status": "succeeded",
                "context": {"output": "RGB(135, 206, 235)"},
                "formula": "moonshot/web-search:latest",
            },
        )

    client = make_sync_client(handler)
    fiber = client.formulas.invoke(
        "moonshot/web-search:latest",
        name="web_search",
        arguments={"query": "sky blue"},
    )
    assert captured == {"name": "web_search", "arguments": '{"query": "sky blue"}'}
    assert fiber.output_str == "RGB(135, 206, 235)"


def test_formula_fiber_prefers_encrypted_output() -> None:
    fiber = FormulaFiber.model_validate(
        {
            "id": "fiber-1",
            "object": "fiber",
            "status": "succeeded",
            "context": {
                "encrypted_output": "----MOONSHOT ENCRYPTED BEGIN----abc----MOONSHOT ENCRYPTED END----",
            },
            "formula": "moonshot/web-search:latest",
        }
    )
    assert fiber.output_str.startswith("----MOONSHOT ENCRYPTED BEGIN----")


def test_formula_fiber_failure_status_returns_error_string() -> None:
    fiber = FormulaFiber.model_validate(
        {
            "id": "f",
            "object": "fiber",
            "status": "failed",
            "context": {"error": "boom"},
        }
    )
    assert fiber.output_str.startswith("Error:")


def test_formulas_load_wraps_tools() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tools"):
            return httpx.Response(200, json=_TOOL_LIST)
        return httpx.Response(
            200,
            json={
                "id": "fiber-1",
                "object": "fiber",
                "status": "succeeded",
                "context": {"output": "ok"},
                "formula": "moonshot/web-search:latest",
            },
        )

    client = make_sync_client(handler)
    [tool] = client.formulas.load("moonshot/web-search:latest")
    assert isinstance(tool, FormulaTool)
    assert tool.name == "web_search"
    assert tool.invoke({"query": "x"}) == "ok"
    td = tool.to_tool_def().model_dump(exclude_none=True, by_alias=True)
    assert td["type"] == "function"
    assert td["function"]["name"] == "web_search"


async def test_async_formulas_load_and_invoke() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tools"):
            return httpx.Response(200, json=_TOOL_LIST)
        return httpx.Response(
            200,
            json={
                "id": "fiber-1",
                "object": "fiber",
                "status": "succeeded",
                "context": {"output": "ok"},
                "formula": "moonshot/web-search:latest",
            },
        )

    client: AsyncKimiClient = make_async_client(handler)
    [tool] = await client.formulas.load("moonshot/web-search:latest")
    assert isinstance(tool, AsyncFormulaTool)
    assert await tool.ainvoke({"query": "x"}) == "ok"
    with pytest.raises(RuntimeError):
        tool.invoke({"query": "x"})
