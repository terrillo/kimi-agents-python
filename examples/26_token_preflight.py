"""Token preflight — estimate before sending."""

from kimi_agents_python import KimiClient, MODEL_SPECS, Model, Session

with KimiClient() as client:
    session = Session(client, model=Model.KIMI_K2_6, system="You are Kimi.")
    draft = "Write me a 500-word essay about the moon."
    n = session.estimated_tokens(draft)
    limit = MODEL_SPECS[Model.KIMI_K2_6].context_length
    print(f"would send ~{n} tokens (context window: {limit})")
    if n < limit // 2:
        reply = session.send(draft)
        print(reply.content[:200], "...")
    else:
        print("skipping — too close to the limit")
