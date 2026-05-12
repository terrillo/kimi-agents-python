"""Runner.run_parallel — multiple independent agents run concurrently.

Total wall time ≈ the slowest single agent call, not the sum of all calls.
Each RunResult carries independent usage and cost — sum them to get totals.

A shared RunContext means one `ctx.cancel.set()` aborts all running agents.
"""

import asyncio
from decimal import Decimal

from kimi_agents_python import AsyncKimiClient, KimiAgent, Model, RunContext, Runner


async def main() -> None:
    async with AsyncKimiClient() as client:
        market_agent = KimiAgent(
            name="market_analyst",
            model=Model.KIMI_K2_6,
            instructions="You are a market analyst. Answer in one sentence.",
        )
        tech_agent = KimiAgent(
            name="tech_writer",
            model=Model.KIMI_K2_6,
            instructions="You are a technical writer. Be concise and accurate.",
        )
        legal_agent = KimiAgent(
            name="legal_summariser",
            model=Model.KIMI_K2_6,
            instructions="You summarise legal concepts clearly for non-lawyers.",
        )

        ctx = RunContext()  # shared — ctx.cancel.set() stops all three

        results = await Runner.run_parallel(
            [
                (market_agent, "What drives AI chip demand in 2025?"),
                (tech_agent, "Explain transformer attention in two sentences."),
                (legal_agent, "What is a non-compete clause?"),
            ],
            client=client,
            context=ctx,
        )

    for r in results:
        print(f"[{r.last_agent.name}]\n{r.final_output}\n")

    total_tokens: int = sum(r.usage.total_tokens for r in results)
    total_cost: Decimal = sum((r.cost_usd for r in results), Decimal("0"))
    print(f"total tokens: {total_tokens}  total cost: ${total_cost:.6f}")


asyncio.run(main())
