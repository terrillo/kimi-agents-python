"""Session — the only multi-turn chat path.

`chat.create()` is single-turn only and refuses payloads with a prior
assistant or tool message (raises `ManualMultiTurnError`). `Session` owns
the message list, echoes `reasoning_content` between turns, and tracks
per-conversation usage and cache stats.
"""

from kimi_agents_python import KimiClient, Model, Session

with KimiClient() as client:
    sess = Session(
        client,
        model=Model.KIMI_K2_6,
        system="You are a terse research assistant.",
        prompt_cache_key="demo-session-18",
    )

    answer = sess.send("List the three largest cloud providers by 2024 revenue.")
    print(f"> turn 1\n  {answer.content}\n")

    answer = sess.send("Now sort that list alphabetically.")
    print(f"> turn 2\n  {answer.content}\n")

    print(f"=== history ({len(sess.history)} messages) ===")
    for m in sess.history:
        body = m.content if isinstance(m.content, str) else "[parts]"
        print(f"  [{m.role}] {body[:60]}")

    print(
        f"\n=== usage ===\n"
        f"  requests:        {sess.usage.requests}\n"
        f"  prompt_tokens:   {sess.usage.prompt_tokens}\n"
        f"  completion_tok:  {sess.usage.completion_tokens}\n"
        f"  total_tokens:    {sess.usage.total_tokens}\n"
        f"  cached_tokens:   {sess.usage.cached_tokens}\n"
        f"  cache hit ratio: {sess.cache_stats.hit_ratio:.1%}"
    )
