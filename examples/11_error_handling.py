"""Catching typed errors — auth, rate-limit subclasses, client-side validation."""

from kimi_agents_python import (
    InvalidAuthenticationError,
    KimiAuthenticationError,
    KimiClient,
    Model,
    RateLimitReachedError,
)

# 1. Server-side typed error: 401 with error.type=invalid_authentication_error
#    surfaces as InvalidAuthenticationError (subclass of KimiAuthenticationError).
with KimiClient(api_key="sk-not-a-real-key") as client:
    try:
        client.chat.create(
            model=Model.KIMI_K2_0905_PREVIEW,
            messages=[{"role": "user", "content": "hi"}],
        )
    except InvalidAuthenticationError as e:
        print(f"Auth error ({e.status_code}) [{e.error_type}]: {e.message}")
    except KimiAuthenticationError as e:
        # Sibling typed class IncorrectAPIKeyError also lands here.
        print(f"Other auth class {type(e).__name__}: {e.message}")
    except RateLimitReachedError as e:
        print(f"Rate-limited: {e.message}")

# 2. Client-side validation runs BEFORE any HTTP call and raises ValueError.
with KimiClient(api_key="sk-not-a-real-key") as client:
    try:
        client.chat.create(
            model=Model.KIMI_K2_6,
            messages=[{"role": "user", "content": "x"}],
            temperature=0.3,  # k2.6 locks temperature at 1.0
        )
    except ValueError as e:
        print(f"\nLocal validation rejected the call: {e}")
