"""Auto-retry — transparent retries for 429 / 5xx / transport errors."""

from kimi_agents_python import KimiClient, Model, RetryConfig

# 1. Default behaviour: every client retries up to 3 times with exponential
#    backoff and jitter. Honors a numeric Retry-After header when the API
#    sends one. Nothing to configure for the common case.
with KimiClient() as client:
    response = client.chat.create(
        model=Model.KIMI_K2_0905_PREVIEW,
        messages=[{"role": "user", "content": "ping"}],
    )
    print(response.choices[0].message.content[:80])

# 2. Custom policy. `retries=` accepts either an int (count) or a RetryConfig
#    for full control over delay shape.
custom = RetryConfig(
    max_retries=5,
    initial_delay=2.0,
    backoff_factor=2.0,
    max_delay=60.0,
    jitter=0.25,
)
with KimiClient(retries=custom) as client:
    response = client.chat.create(
        model=Model.KIMI_K2_0905_PREVIEW,
        messages=[{"role": "user", "content": "ping"}],
    )
    print(response.choices[0].message.content[:80])

# 3. Disable retries entirely (e.g., for tests that want raw errors).
with KimiClient(retries=0) as client:
    response = client.chat.create(
        model=Model.KIMI_K2_0905_PREVIEW,
        messages=[{"role": "user", "content": "ping"}],
    )
    print(response.choices[0].message.content[:80])
