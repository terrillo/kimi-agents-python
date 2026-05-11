# kimi-agents-python

A typed Python client for the [Kimi (Moonshot) API](https://platform.kimi.ai/docs/api/overview), built on `httpx` and `pydantic`. Sync and async clients, streaming, model-aware parameter validation, typed exceptions, auto-retry for transient failures, prompt-cache observability, and file-format checks — all 14 model IDs exposed as a `StrEnum` so you never have to remember the exact string.

## Install

```bash
uv add kimi-agents-python
```

Requires Python 3.14+.

Configure your API key in one of three ways (checked in this order):

1. Pass it explicitly: `KimiClient(api_key="sk-...")`
2. Set `MOONSHOT_API_KEY` in your shell environment
3. Add it to a `.env` file — `python-dotenv` is loaded automatically the first time a client is constructed

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

The async client supports the same `stream=True` flow with `async for`.

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

Plain strings are accepted too — useful when Kimi ships a new model before this library's enum is updated:

```python
client.chat(model="kimi-k2.7-preview", messages=[...])
```

## Model specs and parameter validation

Each model ships with a `ModelSpec` describing its capabilities and the parameter constraints documented at [models-overview](https://platform.kimi.ai/docs/api/models-overview). `chat()` validates kwargs against the spec **before** any HTTP call:

```python
from kimi_agents_python import KimiClient, Model, get_model_spec

spec = get_model_spec(Model.KIMI_K2_6)
print(spec.context_length)     # 262144
print(spec.thinking_support)   # ThinkingSupport.CONFIGURABLE
print(spec.supports_video)     # True

with KimiClient() as client:
    client.chat(
        model=Model.KIMI_K2_6,
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.3,   # raises ValueError — k2.6 locks temperature at 1.0
    )
```

Unknown model strings bypass validation so the client stays usable when the server ships new models.

| Family | Models | Notes |
|---|---|---|
| `kimi-k2.6` / `kimi-k2.5` | `kimi-k2.6`, `kimi-k2.5` | temp/top_p/n/penalties locked; `thinking` configurable; vision + video |
| `kimi-k2` | `kimi-k2-0905-preview`, `kimi-k2-0711-preview`, `kimi-k2-turbo-preview` | flexible params; no thinking |
| `kimi-k2-thinking` | `kimi-k2-thinking`, `kimi-k2-thinking-turbo` | always-on thinking; `temp=1.0` fixed |
| `moonshot-v1` | 8k / 32k / 128k / auto + `-vision-preview` variants | `temp=0.0` default; vision variants accept images |

## Thinking models

Thinking models return a `reasoning_content` field alongside `content`. For `kimi-k2.6` you can also toggle the behaviour via the `thinking` parameter:

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

`keep="all"` preserves prior reasoning across multi-turn conversations (k2.6 only).

## Tool calling

```python
tools = [
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

response = client.chat(
    model=Model.KIMI_K2_0905_PREVIEW,
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
    tools=tools,
)
for call in response.choices[0].message.tool_calls or []:
    print(call.function.name, call.function.arguments)
```

See [`examples/05_tool_calling.py`](examples/05_tool_calling.py) for the full multi-turn loop that feeds tool results back to the model.

## Structured output (JSON schema)

```python
schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "summary", "tags"],
}

response = client.chat(
    model=Model.KIMI_K2_0905_PREVIEW,
    messages=[{"role": "user", "content": "Summarize garbage collection."}],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "Summary", "schema": schema},
    },
)
```

`response_format={"type": "json_object"}` is also accepted for unconstrained JSON.

## Partial mode

Prefill the assistant message to constrain the response shape. The API returns only the *new* tokens — concatenate the prefill yourself:

```python
PREFILL = "{"
response = client.chat(
    model=Model.KIMI_K2_0905_PREVIEW,
    messages=[
        {"role": "user", "content": "List three Python web frameworks as JSON."},
        {"role": "assistant", "content": PREFILL, "partial": True},
    ],
)
body = PREFILL + response.choices[0].message.content
```

## Vision and files

Multimodal models accept `image_url` and `video_url` content parts (base64 or `ms://<file_id>` references):

```python
response = client.chat(
    model=Model.KIMI_K2_6,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ],
)
```

Pre-upload format and size checks ship with the package — useful before sending bytes to the files API:

```python
from kimi_agents_python import (
    FilePurpose,
    IMAGE_FORMATS,
    VIDEO_FORMATS,
    supported_formats,
    validate_file,
)

print(sorted(IMAGE_FORMATS))   # ['gif', 'jpeg', 'jpg', 'png', 'webp']
print(sorted(VIDEO_FORMATS))   # ['3gp', '3gpp', 'avi', 'flv', 'mov', 'mp4', ...]

validate_file("photo.png", FilePurpose.IMAGE)   # ok
validate_file("clip.mp4", FilePurpose.IMAGE)    # raises ValueError
```

| Purpose | Allowed extensions |
|---|---|
| `FilePurpose.IMAGE` | png, jpeg, jpg, webp, gif |
| `FilePurpose.VIDEO` | mp4, mpeg, mpg, mov, avi, x-flv, flv, webm, wmv, 3gpp, 3gp |
| `FilePurpose.FILE_EXTRACT` | 55 formats — pdf, docx, csv, md, source code, etc. |
| `FilePurpose.BATCH` | jsonl |

Constants: `MAX_FILE_BYTES = 100 MiB`, `MAX_TOTAL_BYTES = 10 GiB`, `MAX_FILES = 1000`.

## Helper endpoints

```python
client.list_models()                                            # GET /models
client.estimate_tokens(model=Model.KIMI_K2_6, messages=[...])   # POST /tokenizers/estimate-token-count
client.balance()                                                # GET /users/me/balance
```

## Errors

Exceptions form a two-level hierarchy keyed first on HTTP status, then on the `error.type` string the API returns. Catching either level works:

| Status | Status-level class | Typed `error.type` subclasses |
|---|---|---|
| 400 | `KimiBadRequestError` | `ContentFilterError`, `InvalidRequestError` |
| 401 | `KimiAuthenticationError` | `InvalidAuthenticationError`, `IncorrectAPIKeyError` |
| 403 | `KimiPermissionError` | `PermissionDeniedError` |
| 404 | `KimiNotFoundError` | `ResourceNotFoundError` |
| 429 | `KimiRateLimitError` | `EngineOverloadedError`, `ExceededCurrentQuotaError`, `RateLimitReachedError` |
| 5xx | `KimiServerError` | `ServerErrorResponse`, `UnexpectedOutputError` |

All inherit from `KimiAPIError` → `KimiError`. Each exposes `status_code`, `error_type`, `error_code`, `message`, and the raw response body.

```python
from kimi_agents_python import (
    InvalidAuthenticationError,
    KimiRateLimitError,
    RateLimitReachedError,
)

try:
    client.chat(model=Model.KIMI_K2_6, messages=[...])
except InvalidAuthenticationError as e:
    print(f"Bad key: {e.message}")
except RateLimitReachedError:
    # Or catch the broader KimiRateLimitError to cover all three rate-limit subclasses.
    ...
```

Client-side spec violations raise plain `ValueError` *before* any HTTP call.

## Auto-retry

Both clients retry transient failures (HTTP 429, 5xx, and httpx transport errors) with exponential backoff and jitter. A numeric `Retry-After` header is honored when present. Defaults: 3 retries, 1 s initial delay, 30 s cap.

```python
from kimi_agents_python import KimiClient, RetryConfig

KimiClient()                              # 3 retries (default)
KimiClient(retries=5)                     # bump the count
KimiClient(retries=0)                     # disable
KimiClient(retries=RetryConfig(
    max_retries=5,
    initial_delay=2.0,
    backoff_factor=2.0,
    max_delay=60.0,
    jitter=0.25,
))
```

4xx errors other than 429 (auth, bad request, not found) are **not** retried — they surface immediately as the typed exception class.

## Prompt caching

`kimi-k2.*` models auto-cache prompt prefixes server-side. Pass a stable `prompt_cache_key` (a session id, task id, conversation id) to improve hit rate by routing similar prompts to the same cache shard. Each client tracks cumulative hits in `cache_stats`.

```python
from kimi_agents_python import KimiClient, Model

with KimiClient(prompt_cache_key="user-42-session-7") as client:
    for question in (...):
        client.chat(model=Model.KIMI_K2_6, messages=[
            {"role": "system", "content": SHARED_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ])

    print(client.cache_stats)
    # CacheStats(requests=3, prompt_tokens=2400, cached_tokens=1900)
    print(f"hit rate: {client.cache_stats.hit_ratio:.1%}")  # "79.2%"
```

- A per-call `prompt_cache_key=...` overrides the client default.
- `usage.prompt_tokens_details.cached_tokens` is parsed when present (k2 wire format); the older top-level `usage.cached_tokens` is used as a fallback.
- For streaming, stats only tick if the call asks for usage: `stream_options={"include_usage": True}`.
- `client.cache_stats.reset()` zeros the counters.

## Examples

The [`examples/`](examples/) directory has 14 self-contained scripts, each under 60 lines:

```bash
uv run python examples/01_basic_chat.py
```

| Script | Demonstrates |
|---|---|
| `01_basic_chat.py` | Smallest sync chat call |
| `02_streaming.py` | Token-by-token streaming |
| `03_async_chat.py` | `AsyncKimiClient` + `asyncio.gather` |
| `04_thinking.py` | `reasoning_content` from k2.6 |
| `05_tool_calling.py` | Full tool-result follow-up turn |
| `06_vision.py` | Base64 image input |
| `07_json_schema.py` | Structured output |
| `08_partial_mode.py` | Assistant prefill |
| `09_helpers.py` | `list_models` / `estimate_tokens` / `balance` |
| `10_file_validation.py` | Pre-upload checks (no API call) |
| `11_error_handling.py` | Typed errors + client-side validation |
| `12_kimi_tool_decorator.py` | `@kimi_tool` + `run_tools` auto-loop |
| `13_auto_retry.py` | `RetryConfig` for transient failures |
| `14_prompt_caching.py` | `prompt_cache_key` + `cache_stats` |

## Development

```bash
uv sync --all-groups               # install dev deps
uv run pytest                      # 171 tests, <1s
uv run pytest --cov=kimi_agents_python --cov-report=term-missing
```

## CLI

```bash
uv run kimi-agents-python
```

Prints the package's known model list — useful as a smoke test.
