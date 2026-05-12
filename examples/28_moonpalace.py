"""MoonPalace dev mode — route every request through a local debugging proxy.

Start MoonPalace separately:
    moonpalace start --port 9988
"""

from kimi_agents_python import KimiClient, Model

with KimiClient.with_moonpalace() as client:
    print(f"base_url: {client._base_url}")
    print(f"trace id: {client._auth['X-Msh-Trace-Id']}")
    response = client.chat.create(
        model=Model.KIMI_K2_6,
        messages=[{"role": "user", "content": "Hello, who are you?"}],
    )
    print(response.choices[0].message.content)
