"""chat.prefill() — concatenate the prefill onto the response for you."""

import json

from kimi_agents_python import KimiClient, Model

with KimiClient() as client:
    result = client.chat.prefill(
        model=Model.KIMI_K2_6,
        messages=[
            {"role": "user", "content": "List three Python web frameworks as JSON."}
        ],
        prefill="[",
        max_tokens=200,
    )
    print(json.dumps(json.loads(result.text), indent=2))
