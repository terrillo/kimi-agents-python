from __future__ import annotations

from kimi_agents_python import AsyncKimiClient, KimiClient


def test_with_moonpalace_sets_local_base_url() -> None:
    client = KimiClient.with_moonpalace(api_key="test")
    assert client._base_url == "http://127.0.0.1:9988/v1"
    assert "X-Msh-Trace-Id" in client._auth
    assert len(client._auth["X-Msh-Trace-Id"]) > 0
    client.close()


def test_with_moonpalace_custom_port_and_trace_id() -> None:
    client = KimiClient.with_moonpalace(
        api_key="test", port=4242, host="localhost", trace_id="abc123"
    )
    assert client._base_url == "http://localhost:4242/v1"
    assert client._auth["X-Msh-Trace-Id"] == "abc123"
    client.close()


async def test_async_with_moonpalace() -> None:
    client = AsyncKimiClient.with_moonpalace(api_key="test", trace_id="xyz")
    assert client._base_url == "http://127.0.0.1:9988/v1"
    assert client._auth["X-Msh-Trace-Id"] == "xyz"
    await client.aclose()
