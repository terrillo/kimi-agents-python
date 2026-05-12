"""Multi-agent handoffs — an orchestrator delegates to specialist sub-agents.

Agents listed in `handoffs` are converted to KimiTool instances at run time.
The parent model calls them exactly like any other function; the sub-agent runs
to completion and returns its `final_output` as the tool result.

`result.last_agent` tells you which agent produced the final answer — useful
when a chain of handoffs ends somewhere unexpected.
"""

import asyncio

from kimi_agents_python import AsyncKimiClient, KimiAgent, Model, Runner, kimi_tool


# --- specialist agents -------------------------------------------------------


def _mock_search(query: str) -> list[dict]:
    return [
        {"title": "K2.6 benchmark report", "snippet": "Kimi K2.6 scores 87.3 on MMLU..."},
        {"title": "GPT-5 overview", "snippet": "GPT-5 achieves state-of-the-art on coding..."},
    ]


@kimi_tool(read_only=True)
def web_search_mock(query: str) -> list[dict]:
    """Search the web and return a list of results."""
    return _mock_search(query)


researcher = KimiAgent(
    name="researcher",
    model=Model.KIMI_K2_6,
    instructions=(
        "You find accurate, current information using web_search_mock. "
        "Summarise findings clearly. Cite what you found."
    ),
    tools=[web_search_mock],
)

summariser = KimiAgent(
    name="summariser",
    model=Model.KIMI_K2_6,
    instructions="Condense text into two bullet points. Be ruthlessly concise.",
)


# --- orchestrator ------------------------------------------------------------


orchestrator = KimiAgent(
    name="orchestrator",
    model=Model.KIMI_K2_6,
    instructions=(
        "You coordinate tasks. Delegate research to the researcher agent, "
        "then delegate summarisation to the summariser agent, "
        "then deliver a final answer."
    ),
    handoffs=[researcher, summariser],
)


async def main() -> None:
    async with AsyncKimiClient() as client:
        result = await Runner.run(
            orchestrator,
            "Compare Kimi K2.6 and GPT-5 on reasoning benchmarks.",
            client=client,
        )

    print(result.final_output)
    print(f"\nlast_agent: {result.last_agent.name}")
    print(f"messages in transcript: {len(result.messages)}")
    print(f"cost: ${result.cost_usd:.6f}")


asyncio.run(main())
