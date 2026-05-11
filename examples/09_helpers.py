"""Helper endpoints — list_models, estimate_tokens, balance."""

from kimi_agents_python import KimiClient, Model

with KimiClient() as client:
    print("=== Models ===")
    for info in client.list_models():
        ctx = info.context_length or "?"
        print(f"  {info.id:<40}  context={ctx}")

    print("\n=== Token estimate ===")
    est = client.estimate_tokens(
        model=Model.KIMI_K2_0905_PREVIEW,
        messages=[{"role": "user", "content": "Hello, what can you do?"}],
    )
    print(f"  total tokens: {est.data.total_tokens}")

    print("\n=== Balance ===")
    bal = client.balance()
    print(f"  available: ${bal.data.available_balance:.4f}")
    print(f"  voucher:   ${bal.data.voucher_balance:.4f}")
    print(f"  cash:      ${bal.data.cash_balance:.4f}")
