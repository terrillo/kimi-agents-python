"""Transcript compaction — shrink the per-turn payload, keep the full history.

A `compactor` runs immediately before every `chat._create` in the tool loop.
It receives the full running conversation and returns the message list to
actually send. The loop keeps accumulating the *full* transcript internally,
so `result.messages` is always complete — only the wire payload is rewritten.

Here a tool returns a large document body. Without compaction, every later
turn re-sends every prior body and token cost balloons. The compactor below
elides all but the most recent tool result, so the model still sees the
latest data while old bodies collapse to a short placeholder.
"""

import asyncio
from typing import Annotated

from pydantic import Field

from kimi_agents_python import (
    AsyncKimiClient,
    KimiAgent,
    Model,
    Role,
    Runner,
    kimi_tool,
)

# A stand-in "knowledge base": each fetch returns a deliberately large body.
_DOCS = {
    "solar": "Solar irradiance notes. " + "photovoltaic cell efficiency. " * 400,
    "wind": "Wind turbine notes. " + "rotor swept area and Betz limit. " * 400,
    "hydro": "Hydroelectric notes. " + "head height and flow rate. " * 400,
}


@kimi_tool(read_only=True)
def fetch_document(
    topic: Annotated[str, Field(description="One of: solar, wind, hydro")],
) -> dict:
    """Fetch the full reference document for an energy topic."""
    return {"topic": topic, "body": _DOCS.get(topic, "No such document.")}


def elide_old_tool_results(convo: list[dict]) -> list[dict]:
    """Keep only the most recent tool result in full; collapse earlier ones."""
    last = max(
        (i for i, m in enumerate(convo) if m.get("role") == "tool"), default=-1
    )
    return [
        {**m, "content": "[elided]"} if m.get("role") == "tool" and i != last else m
        for i, m in enumerate(convo)
    ]


async def main() -> None:
    async with AsyncKimiClient() as client:
        agent = KimiAgent(
            name="energy_analyst",
            model=Model.KIMI_K2_6,
            instructions=(
                "Research the user's question by fetching each relevant "
                "document one at a time, then give a concise comparison."
            ),
            tools=[fetch_document],
            compactor=elide_old_tool_results,
            max_steps=8,
        )

        result = await Runner.run(
            agent,
            "Compare solar, wind, and hydro power. Fetch each document.",
            client=client,
        )

        print(result.final_output)

        full_tool_chars = sum(
            len(m.content or "") for m in result.messages if m.role == Role.TOOL
        )
        print(f"\nfull transcript tool-result chars retained: {full_tool_chars:,}")
        print(f"tokens: {result.usage.total_tokens}  cost: ${result.cost_usd:.6f}")


asyncio.run(main())
