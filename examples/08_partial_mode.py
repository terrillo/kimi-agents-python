"""Partial mode — prefill the assistant message to constrain output shape."""

import json

from kimi_agents_python import KimiClient, Model

PREFILL = "{"

with KimiClient() as client:
    response = client.chat.create(
        model=Model.KIMI_K2_0905_PREVIEW,
        messages=[
            {"role": "user", "content": "List three Python web frameworks."},
            {"role": "assistant", "content": PREFILL, "partial": True},
        ],
        max_tokens=200,
    )

# The API returns only the *new* tokens — concatenate the prefill ourselves.
body = PREFILL + response.choices[0].message.content
print(json.dumps(json.loads(body), indent=2))
