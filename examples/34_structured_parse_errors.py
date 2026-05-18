"""chat.parse() across models, with actionable StructuredParseError handling.

json_schema works on every model the client ships, but it's a soft constraint:
models can ramble, so a too-small max_tokens truncates the JSON. Instead of an
opaque pydantic "Invalid JSON: EOF", chat.parse() raises StructuredParseError
telling you what to change. The raw completion is on err.completion.
"""

from pydantic import BaseModel

from kimi_agents_python import KimiClient, Model, StructuredParseError


class City(BaseModel):
    name: str
    country: str
    population_millions: float


with KimiClient() as client:
    # 1. Happy path — typed value out.
    result = client.chat.parse(
        model=Model.KIMI_K2_0905_PREVIEW,
        messages=[{"role": "user", "content": "Structured data for Tokyo."}],
        response_format=City,
        max_tokens=256,
    )
    city = result.parsed
    print(f"{city.name}, {city.country} — {city.population_millions}M")

    # 2. Force a truncation: 8 tokens can't fit the object, so the JSON is
    #    incomplete. The error says to raise max_tokens, not "Invalid JSON".
    try:
        client.chat.parse(
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[{"role": "user", "content": "Structured data for Tokyo."}],
            response_format=City,
            max_tokens=8,
        )
    except StructuredParseError as e:
        print(f"\nParse failed: {e}")
        print(f"finish_reason: {e.completion.choices[0].finish_reason}")
