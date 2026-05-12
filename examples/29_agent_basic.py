"""KimiAgent + Runner — the simplest agent: instructions, one prompt, no tools.

`KimiAgent` holds configuration (model, instructions, tools, guards).
`Runner.run()` executes it and returns a `RunResult` with the final output,
full message transcript, and per-run usage + cost.
"""

import asyncio

from kimi_agents_python import AsyncKimiClient, KimiAgent, Model, Runner


async def main() -> None:
    async with AsyncKimiClient() as client:
        agent = KimiAgent(
            name="assistant",
            model=Model.KIMI_K2_6,
            instructions="You are a terse, precise assistant. Answer in one sentence.",
        )

        result = await Runner.run(
            agent,
            "What is the capital of France?",
            client=client,
        )

        print(result.final_output)
        print(
            f"tokens:  {result.usage.total_tokens} "
            f"(prompt={result.usage.prompt_tokens} "
            f"completion={result.usage.completion_tokens})"
        )
        print(f"cost:    ${result.cost_usd:.6f}")
        print(f"agent:   {result.last_agent.name}")
        print(f"run_id:  {result.usage.requests} request(s)")


asyncio.run(main())
