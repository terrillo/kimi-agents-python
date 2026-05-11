"""Thinking model — read reasoning_content alongside the final answer."""

from kimi_agents_python import KimiClient, Model

with KimiClient() as client:
    response = client.chat(
        model=Model.KIMI_K2_6,
        messages=[
            {"role": "user", "content": "If x*x = 2x + 35 and x > 0, what is x?"}
        ],
        thinking={"type": "enabled"},
        max_tokens=8000,
    )

msg = response.choices[0].message
print("=== Thinking ===")
print(msg.reasoning_content or "(none)")
print("\n=== Answer ===")
print(msg.content)
