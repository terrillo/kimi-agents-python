"""@kimi_tool — function tools with automatic schema + auto tool-call loop."""

from typing import Annotated, Literal

from pydantic import Field

from kimi_agents_python import KimiClient, Model, kimi_tool, run_tools


@kimi_tool
def get_weather(
    city: Annotated[str, Field(description="City name, e.g. 'Tokyo'")],
    units: Literal["c", "f"] = "c",
) -> dict:
    """Get current weather for a city."""
    return {
        "city": city,
        "temperature": 21 if units == "c" else 70,
        "units": units,
        "conditions": "sunny",
    }


@kimi_tool(name_override="lookup_timezone", can_parallel=True)
def timezone_for(city: Annotated[str, Field(description="City name")]) -> str:
    """Return the IANA timezone string for a city."""
    return {"Tokyo": "Asia/Tokyo", "Paris": "Europe/Paris"}.get(city, "UTC")


with KimiClient() as client:
    response = run_tools(
        client,
        model=Model.KIMI_K2_6,
        messages=[
            {
                "role": "user",
                "content": "What's the weather in Tokyo, and what timezone is it?",
            }
        ],
        tools=[get_weather, timezone_for],
    )

print(response.choices[0].message.content)
