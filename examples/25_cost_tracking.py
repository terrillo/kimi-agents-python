"""Per-session cost tracking via MODEL_PRICING."""

from kimi_agents_python import KimiClient, Model, Session

with KimiClient() as client:
    session = Session(client, model=Model.KIMI_K2_6, system="You are Kimi.")
    for q in ("Hi", "What's 2+2?", "Name three rivers."):
        session.send(q)
    print(f"requests:        {session.usage.requests}")
    print(f"prompt_tokens:   {session.usage.prompt_tokens}")
    print(f"completion_tok:  {session.usage.completion_tokens}")
    print(f"cached_tokens:   {session.usage.cached_tokens}")
    print(f"cache hit ratio: {session.cache_stats.hit_ratio:.1%}")
    print(f"cost:            ${session.usage.cost_usd}")
