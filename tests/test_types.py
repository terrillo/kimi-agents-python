from __future__ import annotations

import pytest

from kimi_agents_python import (
    BalanceInfo,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Model,
    ModelList,
    TokenEstimate,
)


def test_chat_request_minimal() -> None:
    req = ChatCompletionRequest.model_validate(
        {"model": Model.KIMI_K2_6, "messages": [{"role": "user", "content": "hi"}]}
    )
    dumped = req.model_dump(exclude_none=True, by_alias=True, mode="json")
    assert dumped == {
        "model": "kimi-k2.6",
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_chat_request_max_tokens_alias() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "kimi-k2.6",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2048,
        }
    )
    assert req.max_completion_tokens == 2048
    dumped = req.model_dump(exclude_none=True, by_alias=True, mode="json")
    assert dumped["max_tokens"] == 2048
    assert "max_completion_tokens" not in dumped


def test_chat_request_thinking_and_tools() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "kimi-k2-thinking",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "enabled", "keep": "all"},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Weather lookup.",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "auto",
        }
    )
    assert req.thinking is not None and req.thinking.type == "enabled"
    assert req.tools is not None and req.tools[0].function.name == "get_weather"
    assert req.tool_choice == "auto"


def test_chat_request_multimodal_content() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "kimi-k2.6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": "ms://file_1"},
                        {"type": "video_url", "video_url": {"url": "ms://file_2"}},
                    ],
                }
            ],
        }
    )
    parts = req.messages[0].content
    assert isinstance(parts, list)
    assert len(parts) == 3
    assert parts[0].type == "text"
    assert parts[1].type == "image_url"
    assert parts[2].type == "video_url"


def test_chat_request_response_format_json_schema() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "kimi-k2.6",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Person",
                    "schema": {"type": "object"},
                },
            },
        }
    )
    rf = req.response_format
    assert rf is not None
    assert rf.type == "json_schema"
    assert rf.json_schema is not None
    assert rf.json_schema.schema_ == {"type": "object"}
    dumped = req.model_dump(exclude_none=True, by_alias=True, mode="json")
    assert dumped["response_format"]["json_schema"]["schema"] == {"type": "object"}


def test_chat_completion_parses_response_with_reasoning() -> None:
    completion = ChatCompletion.model_validate(
        {
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "kimi-k2.6",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "thought",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
                "cached_tokens": 0,
            },
        }
    )
    msg = completion.choices[0].message
    assert msg.content == "answer"
    assert msg.reasoning_content == "thought"
    assert completion.usage is not None
    assert completion.usage.cached_tokens == 0


def test_chat_completion_with_tool_calls() -> None:
    completion = ChatCompletion.model_validate(
        {
            "id": "cmpl-2",
            "object": "chat.completion",
            "created": 2,
            "model": "kimi-k2.6",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "f",
                                    "arguments": '{"x":1}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )
    calls = completion.choices[0].message.tool_calls
    assert calls is not None
    assert calls[0].function.name == "f"
    assert calls[0].function.arguments == '{"x":1}'


def test_chat_completion_chunk() -> None:
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "kimi-k2.6",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hi"},
                    "finish_reason": None,
                }
            ],
        }
    )
    assert chunk.choices[0].delta.content == "Hi"
    assert chunk.choices[0].finish_reason is None


def test_model_list_parses() -> None:
    ml = ModelList.model_validate(
        {
            "object": "list",
            "data": [
                {
                    "id": "kimi-k2.6",
                    "object": "model",
                    "context_length": 256000,
                    "supports_image_in": True,
                    "supports_reasoning": True,
                }
            ],
        }
    )
    assert ml.data[0].id == "kimi-k2.6"
    assert ml.data[0].context_length == 256000


def test_balance_info_parses() -> None:
    b = BalanceInfo.model_validate(
        {
            "code": 0,
            "status": True,
            "scode": "0x0",
            "data": {
                "available_balance": 49.58894,
                "voucher_balance": 46.58893,
                "cash_balance": 3.00001,
            },
        }
    )
    assert b.data.available_balance == pytest.approx(49.58894)


def test_token_estimate_parses() -> None:
    e = TokenEstimate.model_validate({"data": {"total_tokens": 80}})
    assert e.data.total_tokens == 80


def test_extra_response_fields_ignored_gracefully() -> None:
    # Server may add new fields; we must not blow up.
    c = ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 1,
            "model": "kimi-k2.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                    "future_field": "ok",
                }
            ],
            "system_fingerprint": "fp_x",
        }
    )
    assert c.choices[0].message.content == "hi"
