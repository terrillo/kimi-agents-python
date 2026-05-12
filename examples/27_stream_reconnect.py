"""Stream auto-reconnect — survives transport drops via partial prefill."""

from kimi_agents_python import KimiClient, Model

with KimiClient() as client:
    chunks = client.chat.stream_with_reconnect(
        model=Model.KIMI_K2_6,
        messages=[
            {"role": "user", "content": "Tell me a short fairy tale."}
        ],
        max_attempts=5,
        retry_delay=1.0,
    )
    for chunk in chunks:
        if chunk.choices and (delta := chunk.choices[0].delta.content):
            print(delta, end="", flush=True)
    print()
