# kimi-agents-python

![Python versions](https://img.shields.io/pypi/pyversions/kimi-agents-python)

<!-- Monthly downloads -->
![Downloads](https://img.shields.io/pypi/dm/kimi-agents-python)

<!-- License -->
![License](https://img.shields.io/pypi/l/kimi-agents-python)

A typed Python client for the [Kimi (Moonshot) API](https://platform.kimi.ai/docs/api/overview), built on `httpx` and `pydantic`. Sync and async clients with namespaced resources (`client.chat`, `client.files`, `client.batches`, `client.models`, `client.tokenizers`, `client.account`, `client.formulas`), streaming, model-aware parameter validation, typed exceptions, auto-retry, and prompt-cache observability — all 14 model IDs exposed as a `StrEnum` so you never have to remember the exact string. A higher-level `Session` keeps multi-turn state (including `reasoning_content` echo for thinking models), and `KimiAgent` + `Runner` build on top for tool-using agents, handoffs, and parallel execution.

## Install

```bash
uv add kimi-agents-python
```

Requires Python 3.12+.

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
    response = client.chat.create(
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
    for chunk in client.chat.create(
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
        response = await client.chat.create(
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
client.chat.create(model="kimi-k2.7-preview", messages=[...])
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
    client.chat.create(
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
| `kimi-k2-thinking` | `kimi-k2-thinking`, `kimi-k2-thinking-turbo` | always-on thinking; `temp=1.0` fixed; `max_tokens` ≥ 16000 required |
| `moonshot-v1` | 8k / 32k / 128k / auto + `-vision-preview` variants | `temp=0.0` default; vision variants accept images |

## Thinking models

Thinking models return a `reasoning_content` field alongside `content`. For `kimi-k2.6` you can also toggle the behaviour via the `thinking` parameter:

```python
response = client.chat.create(
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

## Multi-turn: `Session`

`chat.create()` is single-turn only. Pass it `[system?, user]` or, for prefill, `[system?, user, assistant(partial=True)]` — anything else (a prior `assistant` message, a `tool` result) raises `ManualMultiTurnError` *before* the HTTP call. This is deliberate: thinking models require `reasoning_content` to be echoed back on every turn, and the library refuses to send a payload that would silently 400.

Use `Session` for any multi-turn conversation. It owns the message list, copies `reasoning_content` automatically, tracks per-conversation cache and token stats, runs tool loops, and supports `fork()` / `checkpoint()` for branching and rollback.

```python
from kimi_agents_python import KimiClient, Model, Session

with KimiClient() as client:
    sess = Session(
        client,
        model=Model.KIMI_K2_6,
        system="You are a research assistant.",
        thinking={"type": "enabled", "keep": "all"},
        prompt_cache_key="user-42-task-7",
    )
    sess.send("Find Q3 revenue for the top 3 cloud vendors.")
    sess.send("Now sort by revenue descending.")   # context preserved automatically

    print(sess.history)         # list[Message]
    print(sess.usage)           # TokenStats — prompt / completion / total / cached
    print(sess.cache_stats)     # CacheStats — per-session, separate from client.cache_stats
```

Streaming, tool loops, branching:

```python
for chunk in sess.stream("Explain the methodology."):
    print(chunk.choices[0].delta.content or "", end="", flush=True)

sess.send("What's the weather?", tools=[get_weather])   # runs the tool loop

cid = sess.checkpoint()                                 # snapshot
sess.send("explore a risky direction")
sess.restore(cid)                                       # rollback

branch = sess.fork()                                    # tree-of-thought
```

`AsyncSession` mirrors the surface against `AsyncKimiClient`. `run_tools` / `arun_tools` are the lower-level alternatives for one-shot tool loops.

## Agents

`KimiAgent` bundles everything needed to run an agent — model, instructions, tools, handoffs, guards — into a single configuration object. `Runner` executes it.

```python
import asyncio
from kimi_agents_python import AsyncKimiClient, KimiAgent, Model, Runner

async def main() -> None:
    async with AsyncKimiClient() as client:
        agent = KimiAgent(
            name="assistant",
            model=Model.KIMI_K2_6,
            instructions="You are a terse, precise assistant.",
        )
        result = await Runner.run(agent, "What is the capital of France?", client=client)
        print(result.final_output)
        print(f"tokens: {result.usage.total_tokens}  cost: ${result.cost_usd:.6f}")

asyncio.run(main())
```

`Runner` is stateless — all execution state lives in the returned `RunResult`:

| Field | Type | Contents |
|---|---|---|
| `final_output` | `str` | The agent's final text answer |
| `messages` | `list[Message]` | Full conversation transcript |
| `last_agent` | `KimiAgent` | Which agent produced the final answer (may differ after handoffs) |
| `usage` | `TokenStats` | Prompt / completion / total / cached tokens + cost |
| `cost_usd` | `Decimal` | Total USD cost for this run |

`Runner.run_sync()` wraps `asyncio.run()` for scripts that can't use `async def`. Do not call it from inside an async context.

### Agent tools

Pass `@kimi_tool`-decorated functions in `tools`. The async runner dispatches tools marked `can_parallel=True` concurrently:

```python
from typing import Annotated
from pydantic import Field
from kimi_agents_python import AsyncKimiClient, KimiAgent, LoopGuards, Model, Runner, kimi_tool

@kimi_tool(read_only=True, can_parallel=True)
def get_weather(city: Annotated[str, Field(description="City name")]) -> dict:
    """Get current weather for a city."""
    return {"city": city, "temp_c": 21, "conditions": "sunny"}

async def main() -> None:
    async with AsyncKimiClient() as client:
        agent = KimiAgent(
            name="weather_bot",
            model=Model.KIMI_K2_6,
            instructions="Answer weather questions using available tools.",
            tools=[get_weather],
            guards=LoopGuards(max_tokens=50_000, repeat_threshold=3),
            max_steps=8,
        )
        result = await Runner.run(agent, "Weather in Tokyo?", client=client)
        print(result.final_output)
```

### Handoffs (multi-agent)

Agents in `handoffs` are automatically converted to tools so the parent model can delegate via function calls. `Runner` wires the client at run time — no manual plumbing required.

```python
researcher = KimiAgent(
    name="researcher",
    model=Model.KIMI_K2_6,
    instructions="Find accurate information. Cite sources.",
    tools=[web_search],
    model_settings={"thinking": {"type": "disabled"}},  # required for web_search
)

orchestrator = KimiAgent(
    name="orchestrator",
    model=Model.KIMI_K2_6,
    instructions="Delegate research tasks to the researcher, then synthesise an answer.",
    handoffs=[researcher],
)

result = await Runner.run(orchestrator, "Compare Kimi K2.6 and GPT-5.", client=client)
print(result.last_agent.name)  # which agent finished the job
```

For explicit control, `handoff(agent, client)` builds the tool directly so you can add it to `tools` yourself:

```python
from kimi_agents_python import handoff

parent = KimiAgent(
    name="parent",
    tools=[handoff(researcher, client=client)],
)
```

### Parallel agents

`Runner.run_parallel()` runs independent `(agent, prompt)` pairs concurrently. Total wall time ≈ the slowest single call:

```python
results = await Runner.run_parallel(
    [
        (market_agent, "What drives AI chip demand in 2025?"),
        (tech_agent, "Explain transformer attention in two sentences."),
        (legal_agent, "What is a non-compete clause?"),
    ],
    client=client,
)

for r in results:
    print(f"[{r.last_agent.name}] {r.final_output}")

total_cost = sum(r.cost_usd for r in results)
```

### Cancellation

Pass a `RunContext` to share cancellation across agents. `ctx.cancel.set()` causes any agent that hasn't started yet to raise `RunCancelledError`:

```python
from kimi_agents_python import RunContext, RunCancelledError

ctx = RunContext()

try:
    results = await Runner.run_parallel(
        [(a, "q1"), (b, "q2")],
        client=client,
        context=ctx,
    )
except RunCancelledError:
    ...

# From another task:
ctx.cancel.set()   # signals all agents sharing this context
```

`ctx.metadata` is a free-form dict for passing caller context (request ids, user ids, etc.) into hooks or tools:

```python
ctx = RunContext(metadata={"request_id": "abc-123", "user": "terrillo"})
```

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

response = client.chat.create(
    model=Model.KIMI_K2_0905_PREVIEW,
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
    tools=tools,
)
for call in response.choices[0].message.tool_calls or []:
    print(call.function.name, call.function.arguments)
```

See [`examples/05_tool_calling.py`](examples/05_tool_calling.py) for the full tool-result follow-up flow driven through `Session`.

### `@kimi_tool` + auto-loop

Wrap a Python function with `@kimi_tool` and `run_tools` / `arun_tools` will drive the chat → tool_calls → result loop to completion for you. The decorator builds the JSON schema from the function signature, takes the first docstring line as the description, and JSON-encodes whatever the function returns.

```python
from typing import Annotated, Literal
from pydantic import Field
from kimi_agents_python import KimiClient, Model, kimi_tool, run_tools

@kimi_tool
def get_weather(
    city: Annotated[str, Field(description="City name, e.g. 'Tokyo'")],
    units: Literal["c", "f"] = "c",
) -> dict:
    """Get current weather for a city."""
    return {"city": city, "temp": 21, "units": units}

with KimiClient() as client:
    result = run_tools(
        client,
        model=Model.KIMI_K2_0905_PREVIEW,
        messages=[{"role": "user", "content": "Weather in Tokyo?"}],
        tools=[get_weather],
        max_steps=5,
    )
print(result.choices[0].message.content)
```

The auto-loop preserves `reasoning_content` across turns — required by `kimi-k2-thinking` per the [multi-step tool calls docs](https://platform.kimi.ai/docs/guide/use-kimi-k2-thinking-model#multi-step-tool-call). See [`examples/15_thinking_tools.py`](examples/15_thinking_tools.py).

### Parallel tool dispatch (`can_parallel`)

When the model returns multiple `tool_calls` in one turn, `arun_tools` partitions them: tools marked `can_parallel=True` (the default) run concurrently via `asyncio.gather`; tools marked `can_parallel=False` run sequentially after the parallel batch. Results are stitched back in the model's original `tool_calls` order, so subsequent turns see a deterministic transcript.

Mark a tool non-parallel when it isn't safe to run alongside its siblings — e.g. it mutates shared state, holds a non-reentrant resource, or hits an API with strict per-tool rate limits.

```python
@kimi_tool                              # can_parallel=True (default)
async def fetch_url(url: str) -> dict:
    ...

@kimi_tool(can_parallel=False)          # always runs sequentially
async def append_to_log(entry: str) -> str:
    ...
```

`can_parallel` is client-side metadata — it is **never** serialised onto the Kimi request body. The synchronous `run_tools` always dispatches sequentially regardless of the flag.

### Loop guards (`LoopGuards`)

`max_steps` catches infinite-iteration loops, but real production loops hit three more subtle failure modes: cost overruns, stuck-in-reads, and the model emitting the same call over and over. Pass an optional `LoopGuards` to `run_tools` / `arun_tools` to catch them:

```python
from kimi_agents_python import LoopGuards, run_tools

run_tools(
    client, model=..., messages=..., tools=[...],
    guards=LoopGuards(
        max_tokens=20_000,        # cumulative usage.total_tokens cap
        read_only_streak=8,       # bail after 8 consecutive read-only calls
        repeat_threshold=3,       # bail when same (name, args) appears 3× in a row
    ),
)
```

Each field is opt-in (`None` disables it). Violations raise dedicated subclasses of `KimiToolLoopError`:

| Field | Exception | Triggered when |
|---|---|---|
| `max_tokens` | `TokenBudgetExceededError` | Cumulative `usage.total_tokens` across the loop crosses the budget |
| `read_only_streak` | `ReadOnlyStreakExceededError` | N consecutive calls to tools marked `@kimi_tool(read_only=True)` with no mutating call in between |
| `repeat_threshold` | `RepeatedToolCallError` | Same `(tool name, JSON-normalized args)` appears N times in a row |

Mark read-only tools at declaration time so the streak guard knows what counts:

```python
@kimi_tool(read_only=True)
def search(query: str) -> dict:
    """Pure lookup — does not mutate anything."""
    ...

@kimi_tool  # default read_only=False — assumed mutating
def write_file(path: str, body: str) -> str:
    ...
```

Catch the base `KimiToolLoopError` to handle every termination reason uniformly, or the specific subclass to react differently (e.g. retry with a smaller budget vs. nudge the model with a hint).

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

response = client.chat.create(
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

Prefill the assistant message to constrain the response shape. The API returns only the *new* tokens — `chat.prefill()` handles the splicing for you:

```python
result = client.chat.prefill(
    model=Model.KIMI_K2_0905_PREVIEW,
    messages=[{"role": "user", "content": "List three Python web frameworks as JSON."}],
    prefill="[",
)
print(result.text)  # already concatenated: "[ ... ]"
```

The raw form is still available — just construct the partial assistant message yourself and concatenate after.

## Platform features

| Feature | Surface |
|---|---|
| Server-side web search (`$web_search`) | `from kimi_agents_python import web_search`; pass in `tools=[...]` and disable thinking |
| Official tool catalog (Formulas) | `client.formulas.load("moonshot/web-search:latest")` returns drop-in tools |
| Structured output | `client.chat.parse(response_format=MyPydanticModel)` → `ParsedChatCompletion[T]` |
| Typed stream events | `for ev in client.chat.stream_events(...)` yields `TextDelta` / `ReasoningDelta` / `ToolCallDelta` / `UsageEvent` / `Done` |
| Cost tracking | `session.usage.cost_usd` accumulates USD per session (uses `MODEL_PRICING`) |
| Token preflight | `session.estimated_tokens("draft user turn")` before sending |
| Stream auto-resume | `client.chat.stream_with_reconnect(...)` resumes via partial prefill on transport drops |
| MoonPalace dev proxy | `KimiClient.with_moonpalace()` flips `base_url` to `http://127.0.0.1:9988/v1` |

## Vision

Multimodal models accept `image_url` and `video_url` content parts (base64 or `ms://<file_id>` references):

```python
response = client.chat.create(
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

The `ms://<file_id>` form references a file uploaded via `client.files.create(...)` — see the [Files](#files) section below.

## Files

Upload, list, fetch extracted content, and delete:

```python
from pathlib import Path
from kimi_agents_python import FilePurpose, KimiClient

with KimiClient() as client:
    f = client.files.create(file=Path("report.pdf"), purpose=FilePurpose.FILE_EXTRACT)
    text = client.files.content(f.id).text   # extracted text

    client.files.list()                       # list[FileObject]
    client.files.retrieve(f.id)               # FileObject
    client.files.delete(f.id)                 # FileDeleted
```

`client.files.create(file=...)` accepts a path (`str` / `Path`) or a `(filename, bytes)` tuple for in-memory uploads. Paths are streamed from disk via `path.open("rb")` rather than loaded into memory. Path uploads with a known `FilePurpose` are checked against `validate_file()` before the HTTP call.

### Pre-upload validation

Format and size checks ship with the package — useful when accepting user-supplied files:

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

The same surface lives under `AsyncKimiClient` — every method is a coroutine.

## Batches

```python
from kimi_agents_python import BatchEndpoint, FilePurpose, KimiClient

with KimiClient() as client:
    input_file = client.files.create(file="batch_input.jsonl", purpose=FilePurpose.BATCH)
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint=BatchEndpoint.CHAT_COMPLETIONS,
        completion_window="24h",       # min "12h", max "7d"
        metadata={"job": "nightly"},
    )

    batch = client.batches.retrieve(batch.id)
    client.batches.list(after="cursor", limit=20)
    client.batches.cancel(batch.id)
```

`batch.status` is a `BatchStatus` (`validating` → `in_progress` → `finalizing` → `completed` | `failed` | `expired` | `cancelled`). When complete, `batch.output_file_id` points at a JSONL result file you fetch with `client.files.content(...)`.

The same surface lives under `AsyncKimiClient` — every method is a coroutine.

## Helper endpoints

```python
client.models.list()                                                # GET /models
client.tokenizers.estimate(model=Model.KIMI_K2_6, messages=[...])   # POST /tokenizers/estimate-token-count
client.account.balance()                                            # GET /users/me/balance
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
    client.chat.create(model=Model.KIMI_K2_6, messages=[...])
except InvalidAuthenticationError as e:
    print(f"Bad key: {e.message}")
except RateLimitReachedError:
    # Or catch the broader KimiRateLimitError to cover all three rate-limit subclasses.
    ...
```

Client-side spec violations raise `ValueError` *before* any HTTP call. Two subclasses are worth catching specifically:

**`ManualMultiTurnError`** fires when `chat.create()` is handed a payload with a prior `assistant` or `tool` message — the multi-turn path is closed at the boundary so thinking-model conversations can't silently 400 by dropping `reasoning_content`. The fix is always the same: switch to `Session`.

```python
from kimi_agents_python import ManualMultiTurnError

try:
    client.chat.create(model=Model.KIMI_K2_6, messages=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"},
    ])
except ManualMultiTurnError:
    ...   # use Session instead — see the Multi-turn section above
```

**`ThinkingIncompatibilityError`** fires when a parameter clashes with thinking mode — currently `tool_choice="required"` while thinking is enabled. Moonshot otherwise responds with a generic 400 that takes a while to decode; this check surfaces the exact cause before the request leaves your machine.

```python
from kimi_agents_python import ThinkingIncompatibilityError

try:
    client.chat.create(
        model=Model.KIMI_K2_6,
        messages=[...],
        thinking={"type": "enabled"},
        tool_choice="required",   # rejected client-side
        tools=[...],
    )
except ThinkingIncompatibilityError as e:
    print(e)  # exact field combination + suggested fix
```

### Tool-loop terminations

`run_tools` / `arun_tools` raise `KimiToolLoopError` (or a subclass) when a budget runs out. The base class catches every termination reason uniformly; the subclasses let you react differently.

| Class | Triggered by |
|---|---|
| `KimiToolLoopError` | `max_steps` exhausted (raised directly), or any of the subclasses below |
| `TokenBudgetExceededError` | `LoopGuards.max_tokens` crossed |
| `ReadOnlyStreakExceededError` | `LoopGuards.read_only_streak` consecutive read-only calls without a mutating call |
| `RepeatedToolCallError` | `LoopGuards.repeat_threshold` consecutive identical calls |

See the [Loop guards](#loop-guards-loopguards) section for configuration.

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
        client.chat.create(model=Model.KIMI_K2_6, messages=[
            {"role": "system", "content": SHARED_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ])

    print(client.cache_stats)
    # CacheStats(requests=3, prompt_tokens=2400, cached_tokens=1900)
    print(f"hit rate: {client.cache_stats.hit_ratio:.1%}")  # "79.2%"
```

- A per-call `prompt_cache_key=...` overrides the client default.
- Per-call hits are available on the response: `completion.cached_tokens` (and `chunk.cached_tokens` for the final streaming chunk). The accessor reads `usage.prompt_tokens_details.cached_tokens` and falls back to the legacy top-level `usage.cached_tokens` for older payloads — so you don't have to remember the precedence yourself.
- For streaming, stats only tick if the call asks for usage: `stream_options={"include_usage": True}`.
- `client.cache_stats.reset()` zeros the counters.

```python
response = client.chat.create(model=Model.KIMI_K2_6, messages=[...])
hit_ratio = response.cached_tokens / response.usage.prompt_tokens
log.info("kimi turn: cached=%d / %d (%.1f%%)",
         response.cached_tokens, response.usage.prompt_tokens, hit_ratio * 100)
```

## Examples

The [`examples/`](examples/) directory has self-contained scripts, each under 60 lines:

```bash
uv run python examples/01_basic_chat.py
```

| Script | Demonstrates |
|---|---|
| `01_basic_chat.py` | Smallest sync chat call |
| `02_streaming.py` | Token-by-token streaming |
| `03_async_chat.py` | `AsyncKimiClient` + `asyncio.gather` |
| `04_thinking.py` | `reasoning_content` from k2.6 |
| `05_tool_calling.py` | Tool-result follow-up turn driven via `Session` |
| `06_vision.py` | Base64 image input |
| `07_json_schema.py` | Structured output |
| `08_partial_mode.py` | Assistant prefill |
| `09_helpers.py` | `models.list` / `tokenizers.estimate` / `account.balance` |
| `10_file_validation.py` | Pre-upload checks (no API call) |
| `11_error_handling.py` | Typed errors + client-side validation |
| `12_kimi_tool_decorator.py` | `@kimi_tool` + `run_tools` auto-loop |
| `13_auto_retry.py` | `RetryConfig` for transient failures |
| `14_prompt_caching.py` | `prompt_cache_key` + `cache_stats` |
| `15_thinking_tools.py` | `kimi-k2-thinking` multi-step tool calls |
| `16_files.py` | `client.files` upload / extract / delete |
| `17_batches.py` | `client.batches` submit, poll, fetch results |
| `18_session_basic.py` | `Session` multi-turn chat + per-session usage |
| `19_session_fork_checkpoint.py` | `Session.fork()` branches + `checkpoint()` / `restore()` rollback |
| `20_web_search.py` | `$web_search` builtin tool driven through `Session` |
| `21_formula_tools.py` | Loading and invoking official Formula tools |
| `22_prefill_helper.py` | `chat.prefill()` for assistant-message scaffolding |
| `23_structured_parse.py` | `chat.parse(response_format=...)` returns a typed value |
| `24_stream_events.py` | Typed stream events (`TextDelta`, `Done`, …) |
| `25_cost_tracking.py` | Per-session `cost_usd` accumulation |
| `26_token_preflight.py` | `session.estimated_tokens(content)` before send |
| `27_stream_reconnect.py` | `chat.stream_with_reconnect(...)` |
| `28_moonpalace.py` | `KimiClient.with_moonpalace()` local debugging |
| `29_agent_basic.py` | `KimiAgent` + `Runner.run()` — simplest agent, no tools |
| `30_agent_tools.py` | Agent with `@kimi_tool` functions, parallel dispatch, `LoopGuards` |
| `31_agent_handoffs.py` | Multi-agent handoffs — orchestrator delegates to specialists |
| `32_agent_parallel.py` | `Runner.run_parallel()` — concurrent agents, cost aggregation |

## Development

```bash
uv sync --all-groups               # install dev deps
uv run pytest                      # 269 tests, <1s
uv run pytest --cov=kimi_agents_python --cov-report=term-missing
```

## CLI

```bash
uv run kimi-agents-python
```

Prints the package's known model list — useful as a smoke test.
