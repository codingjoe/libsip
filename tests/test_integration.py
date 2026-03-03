"""Integration tests requiring network access."""

import asyncio

import pytest

from sip.calls import RegisterProtocol


@pytest.mark.integration
def test_stun_discover__public_stun_server():
    """Discover public address via STUN against stun.l.google.com:19302."""

    async def run() -> tuple[str, int]:
        loop = asyncio.get_running_loop()
        protocol = RegisterProtocol(
            ("stun.l.google.com", 19302),
            "sip:test@stun.l.google.com",
            "test",
            "test",
            stun_server_address=("stun.l.google.com", 19302),
        )
        transport, _ = await loop.create_datagram_endpoint(
            lambda: protocol,
            local_addr=("0.0.0.0", 0),  # noqa: S104 – ephemeral port for outbound STUN discovery
        )
        try:
            return await protocol._stun_discover("stun.l.google.com", 19302)
        finally:
            transport.close()

    public_address = asyncio.run(run())
    host, port = public_address
    assert isinstance(host, str)
    assert len(host) > 0
    assert isinstance(port, int)
    assert 1 <= port <= 65535
