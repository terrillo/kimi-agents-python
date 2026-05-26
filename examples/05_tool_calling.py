"""Tool calling — model invokes a Python function and we feed the result back.

Multi-turn flows (including the tool_call → tool_result follow-up turn) must
go through Session — the raw chat.create() refuses payloads with a prior
assistant or tool message. Session drives the loop and keeps history
consistent.
"""

import json

from kimi_agents_python import KimiClient, Model, Session


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


with KimiClient() as client:
    sess = Session(client, model=Model.KIMI_K2_6, tools=TOOLS)
    first = sess.send("What's the weather in Tokyo?")

    if not first.tool_calls:
        print(first.content)
    else:
        for tc in first.tool_calls:
            result = get_weather(**json.loads(tc.function.arguments))
            sess.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(result),
                }
            )
        final = sess.send()  # no new user turn; just ask for the follow-up
        print(final.content)
