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
| 05 | [`05_tool_calling.py`](05_tool_calling.py) | Function tools + tool-result follow-up turn |
| 06 | [`06_vision.py`](06_vision.py) | Base64 image input (pass a local image path) |
| 07 | [`07_json_schema.py`](07_json_schema.py) | Structured output via `response_format=json_schema` |
| 08 | [`08_partial_mode.py`](08_partial_mode.py) | Prefilling the assistant message |
| 09 | [`09_helpers.py`](09_helpers.py) | `list_models()`, `estimate_tokens()`, `balance()` |
| 10 | [`10_file_validation.py`](10_file_validation.py) | Pre-upload format checks (no API call) |
| 11 | [`11_error_handling.py`](11_error_handling.py) | Typed `error.type` subclasses + client-side validation |

## Notes

- The default model is `kimi-k2-0905-preview` because it accepts flexible parameters. `kimi-k2.6` is used where its features (thinking, vision) are the point.
- `06_vision.py` takes an image path argument: `uv run python examples/06_vision.py path/to/photo.png`.
- `10_file_validation.py` is the only example that doesn't call the API — useful as a sanity-check before adding billing keys.
- These scripts hit the live API and cost real tokens. Run `09_helpers.py` first to check your balance.
