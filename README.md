# kimi-agents-python

A typed Python client for the [Kimi (Moonshot) API](https://platform.kimi.ai/docs/api/overview), built on `httpx` and `pydantic`. Sync and async clients, full streaming support, and every model ID exposed as an enum so you never have to remember the exact string.

## Install

```bash
uv add kimi-agents-python
```

Requires Python 3.14+.

Configure your API key in one of three ways (checked in this order):

1. Pass it explicitly: `KimiClient(api_key="sk-...")`
2. Set `MOONSHOT_API_KEY` in your shell environment
3. Add it to a `.env` file in the project root — `python-dotenv` is loaded automatically the first time a client is constructed.

Copy `.env.example` to `.env` to get started:

```
MOONSHOT_API_KEY=sk-your-key-here
```

## Quickstart

```python
from kimi_agents_python import KimiClient, Model

with KimiClient() as client:
    response = client.chat(
        model=Model.KIMI_K2_6,
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "How do I check if an object is an instance of a class?"},
        ],
    )
    print(response.choices[0].message.content)
```

## Streaming

```python
with KimiClient() as client:
    for chunk in client.chat(
        model=Model.KIMI_K2_6,
        messages=[{"role": "user", "content": "Write a haiku."}],
        stream=True,
    ):
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
```

## Async

```python
import asyncio
from kimi_agents_python import AsyncKimiClient, Model

async def main() -> None:
    async with AsyncKimiClient() as client:
        response = await client.chat(
            model=Model.KIMI_K2_6,
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(response.choices[0].message.content)

asyncio.run(main())
```

## Models

All 14 model IDs are exposed as a `StrEnum`:

```python
from kimi_agents_python import Model, AVAILABLE_MODELS

Model.KIMI_K2_6              # "kimi-k2.6"
Model.KIMI_K2_THINKING       # "kimi-k2-thinking"
Model.MOONSHOT_V1_128K       # "moonshot-v1-128k"

for m in AVAILABLE_MODELS:
    print(m.value)
```

Or fetch the live list from the server:

```python
with KimiClient() as client:
    for info in client.list_models():
        print(info.id, info.context_length)
```

Plain strings are accepted too — useful when Kimi ships a new model before this library's enum is updated:

```python
client.chat(model="kimi-k2.7-preview", messages=[...])
```

## Thinking models

Models like `kimi-k2-thinking`, `kimi-k2.6`, and `kimi-k2.5` return a `reasoning_content` field alongside `content`. Configure thinking explicitly with the `thinking` parameter:

```python
response = client.chat(
    model=Model.KIMI_K2_6,
    messages=[{"role": "user", "content": "Solve: 23 * 47"}],
    thinking={"type": "enabled", "keep": "all"},
    max_tokens=16000,
)
msg = response.choices[0].message
print("thought:", msg.reasoning_content)
print("answer:", msg.content)
```

## Tool calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]

response = client.chat(
    model=Model.KIMI_K2_6,
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
    tools=tools,
)
for call in response.choices[0].message.tool_calls or []:
    print(call.function.name, call.function.arguments)
```

## Helper endpoints

```python
client.list_models()                                            # GET /models
client.estimate_tokens(model=Model.KIMI_K2_6, messages=[...])   # POST /tokenizers/estimate-token-count
client.balance()                                                # GET /users/me/balance
```

## Errors

The client raises typed exceptions per HTTP status:

| Status | Exception                  |
|--------|----------------------------|
| 400    | `KimiBadRequestError`      |
| 401    | `KimiAuthenticationError`  |
| 403    | `KimiPermissionError`      |
| 404    | `KimiNotFoundError`        |
| 429    | `KimiRateLimitError`       |
| 5xx    | `KimiServerError`          |

All inherit from `KimiAPIError` (which inherits from `KimiError`) and expose `status_code`, `error_type`, `error_code`, `message`, and the raw response body.

```python
from kimi_agents_python import KimiClient, KimiRateLimitError

try:
    client.chat(model=Model.KIMI_K2_6, messages=[...])
except KimiRateLimitError as e:
    print(f"Rate limited: {e.message}")
```

## CLI

```bash
uv run kimi-agents-python
```

Prints the package's known model list — useful as a smoke test.

## Reference docs

A full mirror of the upstream Kimi API documentation lives in [`./platform/`](./platform/) for offline reference.
