"""Tests for the SIP asyncio protocol handler."""

from unittest.mock import MagicMock

from sip.messages import Request, Response
from sip.protocol import SIPProtocol


class TestSIPProtocol:
    def test_datagram_received__request(self):
        """Dispatch a received SIP request datagram to request_received."""
        protocol = SIPProtocol()
        protocol.request_received = MagicMock()
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com\r\n"
            b"\r\n"
        )
        addr = ("192.0.2.1", 5060)
        protocol.datagram_received(data, addr)
        protocol.request_received.assert_called_once()
        request, called_addr = protocol.request_received.call_args.args
        assert isinstance(request, Request)
        assert request.method == "INVITE"
        assert called_addr == addr

    def test_datagram_received__response(self):
        """Dispatch a received SIP response datagram to response_received."""
        protocol = SIPProtocol()
        protocol.response_received = MagicMock()
        data = b"SIP/2.0 200 OK\r\nVia: SIP/2.0/UDP pc33.atlanta.com\r\n\r\n"
        addr = ("192.0.2.1", 5060)
        protocol.datagram_received(data, addr)
        protocol.response_received.assert_called_once()
        response, called_addr = protocol.response_received.call_args.args
        assert isinstance(response, Response)
        assert response.status_code == 200
        assert called_addr == addr

    def test_request_received__noop(self):
        """Accept a SIP request without raising an error."""
        protocol = SIPProtocol()
        request = Request(method="OPTIONS", uri="sip:bob@biloxi.com", headers={})
        protocol.request_received(request, ("192.0.2.1", 5060))

    def test_response_received__noop(self):
        """Accept a SIP response without raising an error."""
        protocol = SIPProtocol()
        response = Response(status_code=200, reason="OK", headers={})
        protocol.response_received(response, ("192.0.2.1", 5060))
