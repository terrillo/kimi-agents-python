from __future__ import annotations

import json

import httpx

from kimi_agents_python import BuiltinTool, Model, Session, web_search

from .conftest import completion_body, make_sync_client


def test_web_search_is_builtin_tool() -> None:
    assert isinstance(web_search, BuiltinTool)
    assert web_search.name == "$web_search"


def test_web_search_wire_shape() -> None:
    td = web_search.to_tool_def().model_dump(exclude_none=True, by_alias=True)
    assert td == {"type": "builtin_function", "function": {"name": "$web_search"}}


def test_web_search_invoke_echoes_arguments() -> None:
    args = {"query": "moon", "usage": {"total_tokens": 42}}
    assert json.loads(web_search.invoke(args)) == args
    assert web_search.invoke('{"q": "x"}') == '{"q": "x"}'


def test_builtin_tool_in_chat_request_body() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode()))
        return httpx.Response(200, json=completion_body("hi"))

    client = make_sync_client(handler)
    client.chat.create(
        model=Model.KIMI_K2_6,
        messages=[{"role": "user", "content": "what's new?"}],
        tools=[web_search],
        thinking={"type": "disabled"},
    )
    assert captured["tools"] == [
        {"type": "builtin_function", "function": {"name": "$web_search"}}
    ]


def test_builtin_tool_dispatches_in_session_loop() -> None:
    """Round 1: model returns tool_calls. Round 2: model returns final answer."""
    rounds: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        rounds.append(request)
        if len(rounds) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": Model.KIMI_K2_6.value,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "ws:0",
                                        "type": "function",
                                        "function": {
                                            "name": "$web_search",
                                            "arguments": '{"query":"moon"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 5,
                        "total_tokens": 10,
                    },
                },
            )
        return httpx.Response(200, json=completion_body("done"))

    client = make_sync_client(handler)
    session = Session(client, model=Model.KIMI_K2_6, system="sys")
    msg = session.send(
        "search the moon",
        tools=[web_search],
        thinking={"type": "disabled"},
    )
    assert msg.content == "done"
    second_body = json.loads(rounds[1].content.decode())
    tool_msg = next(m for m in second_body["messages"] if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "ws:0"
    assert json.loads(tool_msg["content"]) == {"query": "moon"}
