"""Multi-step tool calls with kimi-k2-thinking — `run_tools` preserves reasoning across turns."""

from typing import Annotated, Literal

from pydantic import Field

from kimi_agents_python import KimiClient, Model, kimi_tool, run_tools


@kimi_tool
def get_weather(
    city: Annotated[str, Field(description="City name, e.g. 'Tokyo'")],
    units: Literal["c", "f"] = "c",
) -> dict:
    """Get current weather for a city."""
    return {"city": city, "temp": 21 if units == "c" else 70, "units": units}


@kimi_tool
def get_time(
    city: Annotated[str, Field(description="City name")],
) -> dict:
    """Get local time for a city."""
    return {"city": city, "time": "14:30"}


with KimiClient() as client:
    result = run_tools(
        client,
        model=Model.KIMI_K2_THINKING,
        messages=[
            {
                "role": "user",
                "content": "What's the weather and local time in Tokyo? "
                "Decide if I should go for a walk.",
            }
        ],
        tools=[get_weather, get_time],
        max_tokens=16000,
        max_steps=8,
    )

    msg = result.choices[0].message
    if msg.reasoning_content:
        print("=== final reasoning ===")
        print(msg.reasoning_content)
    print("\n=== answer ===")
    print(msg.content)
