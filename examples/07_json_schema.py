"""Structured output — response_format with a JSON schema."""

import json

from kimi_agents_python import KimiClient, Model

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "summary", "tags"],
}

with KimiClient() as client:
    response = client.chat.create(
        model=Model.KIMI_K2_6,
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize the concept of garbage collection for a CS undergrad. "
                    "Return JSON matching the schema."
                ),
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "Summary", "schema": SCHEMA},
        },
        max_tokens=400,
    )

print(json.dumps(json.loads(response.choices[0].message.content), indent=2))
