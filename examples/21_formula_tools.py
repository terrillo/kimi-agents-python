"""Official Formula API tools — mount moonshot/web-search via Session."""

from kimi_agents_python import KimiClient, Model, Session

with KimiClient() as client:
    tools = client.formulas.load("moonshot/web-search:latest")
    session = Session(client, model=Model.KIMI_K2_6, system="You are Kimi.")
    answer = session.send(
        "What is the RGB value of sky blue?",
        tools=tools,
        max_steps=3,
    )
    print(answer.content)
