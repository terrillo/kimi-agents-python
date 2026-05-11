"""Tool calling — model invokes a Python function, we feed back the result."""

import json

from kimi_agents_python import KimiClient, Model


def get_weather(city: str) -> dict:
    return {"city": city, "temperature_c": 21, "conditions": "sunny"}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


messages: list[dict] = [{"role": "user", "content": "What's the weather in Tokyo?"}]

with KimiClient() as client:
    first = client.chat(model=Model.KIMI_K2_0905_PREVIEW, messages=messages, tools=TOOLS)
    assistant = first.choices[0].message

    if not assistant.tool_calls:
        print(assistant.content)
    else:
        messages.append(
            {
                "role": "assistant",
                "content": assistant.content,
                "tool_calls": [tc.model_dump() for tc in assistant.tool_calls],
            }
        )
        for tc in assistant.tool_calls:
            result = get_weather(**json.loads(tc.function.arguments))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(result),
                }
            )
        final = client.chat(
            model=Model.KIMI_K2_0905_PREVIEW, messages=messages, tools=TOOLS
        )
        print(final.choices[0].message.content)
