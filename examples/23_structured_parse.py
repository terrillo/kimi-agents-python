"""chat.parse() — Pydantic model in, typed value out."""

from pydantic import BaseModel

from kimi_agents_python import KimiClient, Model


class Summary(BaseModel):
    title: str
    bullets: list[str]


with KimiClient() as client:
    result = client.chat.parse(
        model=Model.KIMI_K2_6,
        messages=[{"role": "user", "content": "Summarize garbage collection."}],
        response_format=Summary,
    )
    print(result.parsed.title)
    for b in result.parsed.bullets:
        print(f"  - {b}")
