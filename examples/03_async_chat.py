"""Async client — fan out multiple requests concurrently."""

import asyncio

from kimi_agents_python import AsyncKimiClient, Model

QUESTIONS = [
    "What is a generator?",
    "What is a coroutine?",
    "What is a context manager?",
]


async def ask(client: AsyncKimiClient, q: str) -> str:
    r = await client.chat.create(
        model=Model.KIMI_K2_0905_PREVIEW,
        messages=[{"role": "user", "content": f"{q} Answer in one sentence."}],
        max_tokens=80,
    )
    return r.choices[0].message.content


async def main() -> None:
    async with AsyncKimiClient() as client:
        answers = await asyncio.gather(*(ask(client, q) for q in QUESTIONS))
    for q, a in zip(QUESTIONS, answers):
        print(f"Q: {q}\nA: {a}\n")


asyncio.run(main())
