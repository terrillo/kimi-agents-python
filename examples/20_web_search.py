"""$web_search builtin tool — Moonshot runs the search server-side."""

from kimi_agents_python import KimiClient, Model, Session, web_search

with KimiClient() as client:
    session = Session(
        client,
        model=Model.KIMI_K2_6,
        system="You are Kimi.",
        # $web_search requires thinking to be disabled.
        thinking={"type": "disabled"},
    )
    answer = session.send(
        "Search for Moonshot AI Context Caching and tell me what it is.",
        tools=[web_search],
        max_steps=3,
    )
    print(answer.content)
    print(f"\nprompt_tokens: {session.usage.prompt_tokens}")
    print(f"cost so far:   ${session.usage.cost_usd}")
