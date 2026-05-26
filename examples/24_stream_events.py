"""Typed stream events — TextDelta / ReasoningDelta / Done / UsageEvent."""

from kimi_agents_python import (
    Done,
    KimiClient,
    Model,
    ReasoningDelta,
    TextDelta,
    UsageEvent,
)

with KimiClient() as client:
    events = client.chat.stream_events(
        model=Model.KIMI_K2_6,
        messages=[{"role": "user", "content": "Explain MoE in one sentence."}],
        thinking={"type": "enabled", "keep": "all"},
        stream_options={"include_usage": True},
    )
    for ev in events:
        match ev:
            case TextDelta(content=c):
                print(c, end="", flush=True)
            case ReasoningDelta(content=c):
                print(f"\n[think] {c}", end="", flush=True)
            case UsageEvent(usage=u):
                print(f"\n[usage] {u.total_tokens} tokens")
            case Done(finish_reason=fr):
                print(f"\n[done] {fr}")
