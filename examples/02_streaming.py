"""Streaming chat — print tokens as they arrive."""

from kimi_agents_python import KimiClient, Model

with KimiClient() as client:
    stream = client.chat.create(
        model=Model.KIMI_K2_6,
        messages=[{"role": "user", "content": "Write a haiku about recursion."}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
print()
