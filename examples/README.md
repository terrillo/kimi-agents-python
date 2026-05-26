# Examples

Self-contained scripts demonstrating the `kimi-agents-python` surface. Each one is short and runnable with:

```bash
uv run python examples/<file>.py
```

## Setup

Set `MOONSHOT_API_KEY` in your environment or `.env` at the repo root (see `.env.example`). Get a key at https://platform.kimi.ai/console/api-keys.

## Index

| # | Script | Demonstrates |
|---|---|---|
| 01 | [`01_basic_chat.py`](01_basic_chat.py) | Smallest sync chat call |
| 02 | [`02_streaming.py`](02_streaming.py) | Token-by-token streaming |
| 03 | [`03_async_chat.py`](03_async_chat.py) | `AsyncKimiClient` + `asyncio.gather` fan-out |
| 04 | [`04_thinking.py`](04_thinking.py) | `kimi-k2.6` with `thinking={"type":"enabled"}` and `reasoning_content` |
| 05 | [`05_tool_calling.py`](05_tool_calling.py) | Function tools + tool-result follow-up via `Session` |
| 06 | [`06_vision.py`](06_vision.py) | Base64 image input (pass a local image path) |
| 07 | [`07_json_schema.py`](07_json_schema.py) | Structured output via `response_format=json_schema` |
| 08 | [`08_partial_mode.py`](08_partial_mode.py) | Prefilling the assistant message |
| 09 | [`09_helpers.py`](09_helpers.py) | `client.models.list()`, `client.tokenizers.estimate()`, `client.account.balance()` |
| 10 | [`10_file_validation.py`](10_file_validation.py) | Pre-upload format checks (no API call) |
| 11 | [`11_error_handling.py`](11_error_handling.py) | Typed `error.type` subclasses + client-side validation |
| 12 | [`12_kimi_tool_decorator.py`](12_kimi_tool_decorator.py) | `@kimi_tool` + `run_tools` auto-loop |
| 13 | [`13_auto_retry.py`](13_auto_retry.py) | Default + custom `RetryConfig` for 429 / 5xx |
| 14 | [`14_prompt_caching.py`](14_prompt_caching.py) | `prompt_cache_key` default + `cache_stats` |
| 15 | [`15_thinking_tools.py`](15_thinking_tools.py) | `kimi-k2.6` thinking-enabled multi-step tool calls (`run_tools` preserves `reasoning_content`) |
| 16 | [`16_files.py`](16_files.py) | `client.files` upload, extract content, chat over it, delete |
| 17 | [`17_batches.py`](17_batches.py) | `client.batches` create + poll + fetch JSONL results |
| 18 | [`18_session_basic.py`](18_session_basic.py) | `Session` multi-turn chat — auto-echoes `reasoning_content`, tracks per-session usage |
| 19 | [`19_session_fork_checkpoint.py`](19_session_fork_checkpoint.py) | `Session.fork()` for branching + `checkpoint()` / `restore()` for rollback |
| 20 | [`20_web_search.py`](20_web_search.py) | `$web_search` builtin tool |
| 21 | [`21_formula_tools.py`](21_formula_tools.py) | Official Formula tools |
| 22 | [`22_prefill_helper.py`](22_prefill_helper.py) | `chat.prefill()` assistant-message scaffolding |
| 23 | [`23_structured_parse.py`](23_structured_parse.py) | `chat.parse(response_format=...)` typed structured output |
| 24 | [`24_stream_events.py`](24_stream_events.py) | Typed stream events (`TextDelta`, `ReasoningDelta`, `Done`, …) |
| 25 | [`25_cost_tracking.py`](25_cost_tracking.py) | Per-session `cost_usd` accumulation |
| 26 | [`26_token_preflight.py`](26_token_preflight.py) | `session.estimated_tokens()` before send |
| 27 | [`27_stream_reconnect.py`](27_stream_reconnect.py) | `chat.stream_with_reconnect()` auto-resume |
| 28 | [`28_moonpalace.py`](28_moonpalace.py) | `KimiClient.with_moonpalace()` local debugging proxy |
| **29** | **[`29_agent_basic.py`](29_agent_basic.py)** | **`KimiAgent` + `Runner.run()` — simplest agent, no tools** |
| **30** | **[`30_agent_tools.py`](30_agent_tools.py)** | **`KimiAgent` with `@kimi_tool` functions, parallel dispatch, `LoopGuards`** |
| **31** | **[`31_agent_handoffs.py`](31_agent_handoffs.py)** | **Multi-agent handoffs — orchestrator delegates to specialist sub-agents** |
| **32** | **[`32_agent_parallel.py`](32_agent_parallel.py)** | **`Runner.run_parallel()` — concurrent agents, shared `RunContext`, cost aggregation** |
| **33** | **[`33_agent_compaction.py`](33_agent_compaction.py)** | **`KimiAgent.compactor` — shrink the per-turn payload, keep the full transcript** |
| 34 | [`34_structured_parse_errors.py`](34_structured_parse_errors.py) | `chat.parse()` with `StructuredParseError` handling for truncated output |

## Notes

- All examples use `kimi-k2.6` — the current production model. Its parameters (`temperature`, `top_p`, `n`, `presence_penalty`, `frequency_penalty`) are locked at the server-recommended defaults; `11_error_handling.py` demonstrates the client-side validation that surfaces this before the wire.
- `06_vision.py` takes an image path argument: `uv run python examples/06_vision.py path/to/photo.png`.
- `10_file_validation.py` is the only example that doesn't call the API — useful as a sanity-check before adding billing keys.
- These scripts hit the live API and cost real tokens. Run `09_helpers.py` first to check your balance.
- Agent examples (29–32) all use `AsyncKimiClient` — `Runner` is async-first. `Runner.run_sync()` wraps `asyncio.run()` for scripts that can't use `async def main()`.
