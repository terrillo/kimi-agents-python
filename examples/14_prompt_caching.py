"""Prompt caching — set a stable cache key and observe cache_stats."""

from kimi_agents_python import KimiClient, Model

# kimi-k2.* models auto-cache prompt prefixes. Setting a stable prompt_cache_key
# (a session id, task id, etc.) routes similar requests to the same cache shard
# and improves hit rate. Pass it once at client construction and every chat()
# call inherits it.
with KimiClient(prompt_cache_key="demo-session-1") as client:
    SHARED_SYSTEM = (
        "You are a senior Python reviewer. Reply with one short sentence per turn."
    )

    for question in (
        "What's a good rule for naming test files?",
        "And for module-level constants?",
        "What about private helpers?",
    ):
        response = client.chat(
            model=Model.KIMI_K2_6,
            messages=[
                {"role": "system", "content": SHARED_SYSTEM},
                {"role": "user", "content": question},
            ],
        )
        print(f"> {question}\n  {response.choices[0].message.content}\n")

    stats = client.cache_stats
    print(
        f"=== cache stats ===\n"
        f"  requests:      {stats.requests}\n"
        f"  prompt_tokens: {stats.prompt_tokens}\n"
        f"  cached_tokens: {stats.cached_tokens}\n"
        f"  hit_ratio:     {stats.hit_ratio:.1%}"
    )

# A per-call prompt_cache_key overrides the default — useful when one client
# instance multiplexes requests for many sessions.
with KimiClient(prompt_cache_key="default") as client:
    client.chat(
        model=Model.KIMI_K2_6,
        messages=[{"role": "user", "content": "hi"}],
        prompt_cache_key="user-42-session-7",
    )
