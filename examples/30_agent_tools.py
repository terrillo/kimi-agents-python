"""KimiAgent with @kimi_tool functions, parallel dispatch, and LoopGuards.

Tools marked `can_parallel=True` (default) run concurrently via asyncio.gather
when the model calls several at once. `LoopGuards` cap cost and detect stuck
loops before they burn budget.
"""

import asyncio
from typing import Annotated

from pydantic import Field

from kimi_agents_python import (
    AsyncKimiClient,
    KimiAgent,
    LoopGuards,
    Model,
    Runner,
    kimi_tool,
)


@kimi_tool(read_only=True, can_parallel=True)
def get_weather(
    city: Annotated[str, Field(description="City name, e.g. 'Tokyo'")],
) -> dict:
    """Get current weather for a city."""
    return {"city": city, "temp_c": 21, "conditions": "sunny"}


@kimi_tool(read_only=True, can_parallel=True)
def get_population(
    city: Annotated[str, Field(description="City name")],
) -> dict:
    """Return the city's estimated population."""
    populations = {"Tokyo": 13_960_000, "Paris": 2_100_000, "London": 8_900_000}
    return {"city": city, "population": populations.get(city, 1_000_000)}


async def main() -> None:
    async with AsyncKimiClient() as client:
        agent = KimiAgent(
            name="city_analyst",
            model=Model.KIMI_K2_6,
            instructions="Answer questions about cities using available tools. Be concise.",
            tools=[get_weather, get_population],
            guards=LoopGuards(
                max_tokens=50_000,    # abort if the loop burns too many tokens
                repeat_threshold=3,   # abort if the same call repeats 3× in a row
            ),
            max_steps=8,
        )

        result = await Runner.run(
            agent,
            "What's the weather and population of Tokyo and Paris?",
            client=client,
        )

        print(result.final_output)
        print(f"\ntokens: {result.usage.total_tokens}  cost: ${result.cost_usd:.6f}")


asyncio.run(main())
