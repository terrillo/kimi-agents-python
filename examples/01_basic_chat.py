"""Smallest possible sync chat completion."""

from kimi_agents_python import KimiClient, Model

with KimiClient() as client:
    response = client.chat.create(
        model=Model.KIMI_K2_6,
        messages=[
            {"role": "system", "content": "You are a concise coding assistant."},
            {"role": "user", "content": "Explain what __slots__ does in Python."},
        ],
        max_tokens=300,
    )

print(response.choices[0].message.content)
print(f"\n[tokens: {response.usage.total_tokens}]")
