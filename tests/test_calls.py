"""Tests for SIP call handling."""

import asyncio
from unittest.mock import MagicMock

import pytest
from sip.calls import IncomingCall, IncomingCallProtocol, RTPProtocol
from sip.messages import Message, Request


def make_invite(headers: dict | None = None) -> Request:
    """Return an INVITE request with default headers."""
    return Request(
        method="INVITE",
        uri="sip:alice@atlanta.com",
        headers={
            "Via": "SIP/2.0/UDP pc33.atlanta.com",
            "To": "sip:alice@atlanta.com",
            "From": "sip:bob@biloxi.com",
            "Call-ID": "1234@pc33",
            "CSeq": "1 INVITE",
            **(headers or {}),
        },
    )


def make_call(transport: MagicMock | None = None) -> IncomingCall:
    """Return an IncomingCall with a mock transport."""
    return IncomingCall(
        make_invite(),
        ("192.0.2.1", 5060),
        transport or MagicMock(),
    )


class TestRTPProtocol:
    def test_datagram_received__forwards_audio(self):
        """Strip RTP header and forward audio payload via handle."""
        received = []

        class ConcreteRTP(RTPProtocol):
            def handle(self, audio: bytes) -> None:
                received.append(audio)

        protocol = ConcreteRTP()
        rtp_packet = b"\x80\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00" + b"audio"
        protocol.datagram_received(rtp_packet, ("192.0.2.1", 5004))
        assert received == [b"audio"]

    def test_datagram_received__skips_short_packet(self):
        """Skip packets shorter than the minimum RTP header size."""
        received = []

        class ConcreteRTP(RTPProtocol):
            def handle(self, audio: bytes) -> None:
                received.append(audio)

        ConcreteRTP().datagram_received(b"\x80\x00", ("192.0.2.1", 5004))
        assert received == []

    def test_datagram_received__skips_exact_header_size(self):
        """Skip packets that contain only an RTP header with no audio payload."""
        received = []

        class ConcreteRTP(RTPProtocol):
            def handle(self, audio: bytes) -> None:
                received.append(audio)

        ConcreteRTP().datagram_received(b"\x80" * 12, ("192.0.2.1", 5004))
        assert received == []

    def test_handle__returns_not_implemented(self):
        """Return NotImplemented for unhandled audio data."""
        assert RTPProtocol().handle(b"audio") is NotImplemented


class TestIncomingCall:
    def test_caller__returns_from_header(self):
        """Return the caller's SIP address from the From header."""
        assert make_call().caller == "sip:bob@biloxi.com"

    def test_caller__missing_header(self):
        """Return an empty string when the From header is absent."""
        call = IncomingCall(
            Request(method="INVITE", uri="sip:alice@atlanta.com"),
            ("192.0.2.1", 5060),
            MagicMock(),
        )
        assert call.caller == ""

    def test_handle__returns_not_implemented(self):
        """Return NotImplemented for unhandled audio data."""
        assert make_call().handle(b"audio") is NotImplemented

    def test_reject__sends_busy_here_by_default(self):
        """Send a 486 Busy Here response when no status code is given."""
        transport = MagicMock()
        make_call(transport).reject()
        transport.sendto.assert_called_once()
        sent_bytes, addr = transport.sendto.call_args[0]
        response = Message.parse(sent_bytes)
        assert response.status_code == 486
        assert response.reason == "Busy Here"
        assert addr == ("192.0.2.1", 5060)

    def test_reject__custom_status(self):
        """Send the specified status code and reason."""
        transport = MagicMock()
        make_call(transport).reject(status_code=603, reason="Decline")
        sent_bytes, _ = transport.sendto.call_args[0]
        response = Message.parse(sent_bytes)
        assert response.status_code == 603
        assert response.reason == "Decline"

    def test_reject__copies_dialog_headers(self):
        """Copy Via, To, From, Call-ID, and CSeq headers into the response."""
        transport = MagicMock()
        make_call(transport).reject()
        sent_bytes, _ = transport.sendto.call_args[0]
        response = Message.parse(sent_bytes)
        assert response.headers["Via"] == "SIP/2.0/UDP pc33.atlanta.com"
        assert response.headers["To"] == "sip:alice@atlanta.com"
        assert response.headers["From"] == "sip:bob@biloxi.com"
        assert response.headers["Call-ID"] == "1234@pc33"
        assert response.headers["CSeq"] == "1 INVITE"

    def test_answer__sends_200_ok(self):
        """Send a 200 OK response with an SDP body when answering."""

        async def run() -> None:
            transport = MagicMock()
            await make_call(transport).answer()
            transport.sendto.assert_called_once()
            sent_bytes, addr = transport.sendto.call_args[0]
            response = Message.parse(sent_bytes)
            assert response.status_code == 200
            assert response.reason == "OK"
            assert addr == ("192.0.2.1", 5060)

        asyncio.run(run())

    def test_answer__sdp_contains_audio_line(self):
        """Include an audio media line in the SDP body of the 200 OK."""

        async def run() -> None:
            transport = MagicMock()
            await make_call(transport).answer()
            sent_bytes, _ = transport.sendto.call_args[0]
            response = Message.parse(sent_bytes)
            assert b"m=audio" in response.body
            assert b"RTP/AVP" in response.body

        asyncio.run(run())

    def test_answer__copies_dialog_headers(self):
        """Copy Via, To, From, Call-ID, and CSeq headers into the 200 OK."""

        async def run() -> None:
            transport = MagicMock()
            await make_call(transport).answer()
            sent_bytes, _ = transport.sendto.call_args[0]
            response = Message.parse(sent_bytes)
            assert response.headers["Via"] == "SIP/2.0/UDP pc33.atlanta.com"
            assert response.headers["To"] == "sip:alice@atlanta.com"
            assert response.headers["From"] == "sip:bob@biloxi.com"
            assert response.headers["Call-ID"] == "1234@pc33"
            assert response.headers["CSeq"] == "1 INVITE"

        asyncio.run(run())

    def test_answer__rtp_receives_audio(self):
        """Deliver audio from RTP packets to the call's handle method via the RTP socket."""
        received_audio = []

        class AudioCapture(IncomingCall):
            def handle(self, audio: bytes) -> None:
                received_audio.append(audio)

        async def run() -> None:
            transport = MagicMock()
            call = AudioCapture(make_invite(), ("192.0.2.1", 5060), transport)
            await call.answer()
            sent_bytes, _ = transport.sendto.call_args[0]
            response = Message.parse(sent_bytes)

            sdp_line = next(
                line
                for line in response.body.decode().splitlines()
                if line.startswith("m=audio")
            )
            rtp_port = int(sdp_line.split()[1])

            loop = asyncio.get_running_loop()
            send_transport, _ = await loop.create_datagram_endpoint(
                asyncio.DatagramProtocol,
                remote_addr=("127.0.0.1", rtp_port),
            )
            rtp_packet = b"\x80\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00audio"
            send_transport.sendto(rtp_packet)
            await asyncio.sleep(0.05)
            send_transport.close()

        asyncio.run(run())
        assert received_audio == [b"audio"]

    def test_answer__rtp_receives_multiple_packets(self):
        """Call handle with each RTP payload when multiple packets arrive."""
        received_audio = []

        class AudioCapture(IncomingCall):
            def handle(self, audio: bytes) -> None:
                received_audio.append(audio)

        async def run() -> None:
            transport = MagicMock()
            call = AudioCapture(make_invite(), ("192.0.2.1", 5060), transport)
            await call.answer()
            sent_bytes, _ = transport.sendto.call_args[0]
            response = Message.parse(sent_bytes)
            sdp_line = next(
                line
                for line in response.body.decode().splitlines()
                if line.startswith("m=audio")
            )
            rtp_port = int(sdp_line.split()[1])

            loop = asyncio.get_running_loop()
            send_transport, _ = await loop.create_datagram_endpoint(
                asyncio.DatagramProtocol,
                remote_addr=("127.0.0.1", rtp_port),
            )
            header = b"\x80\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00"
            send_transport.sendto(header + b"chunk1")
            send_transport.sendto(header + b"chunk2")
            await asyncio.sleep(0.05)
            send_transport.close()

        asyncio.run(run())
        assert received_audio == [b"chunk1", b"chunk2"]

    @pytest.mark.parametrize(
        "extra_header",
        ["X-Custom"],
    )
    def test_reject__excludes_extra_headers(self, extra_header):
        """Exclude non-dialog headers from the reject response."""
        transport = MagicMock()
        call = IncomingCall(
            make_invite({extra_header: "value"}),
            ("192.0.2.1", 5060),
            transport,
        )
        call.reject()
        sent_bytes, _ = transport.sendto.call_args[0]
        response = Message.parse(sent_bytes)
        assert extra_header not in response.headers


class TestIncomingCallProtocol:
    def test_request_received__invite__calls_invite_received(self):
        """Dispatch an INVITE request to invite_received with an IncomingCall."""
        calls = []

        class ConcreteProtocol(IncomingCallProtocol):
            def invite_received(self, call, addr):
                calls.append((call, addr))

        protocol = ConcreteProtocol()
        protocol._transport = MagicMock()
        request = Request(
            method="INVITE",
            uri="sip:alice@atlanta.com",
            headers={"From": "sip:bob@biloxi.com"},
        )
        addr = ("192.0.2.1", 5060)
        protocol.request_received(request, addr)
        assert len(calls) == 1
        call, called_addr = calls[0]
        assert isinstance(call, IncomingCall)
        assert call.caller == "sip:bob@biloxi.com"
        assert called_addr == addr

    def test_request_received__non_invite__returns_not_implemented(self):
        """Return NotImplemented for non-INVITE requests."""
        protocol = IncomingCallProtocol()
        request = Request(method="OPTIONS", uri="sip:alice@atlanta.com")
        assert protocol.request_received(request, ("192.0.2.1", 5060)) is NotImplemented

    def test_invite_received__returns_not_implemented(self):
        """Return NotImplemented for unhandled incoming calls."""
        protocol = IncomingCallProtocol()
        assert protocol.invite_received(MagicMock(), ("192.0.2.1", 5060)) is NotImplemented

    def test_create_call__returns_incoming_call(self):
        """Return an IncomingCall bound to the SIP transport."""
        protocol = IncomingCallProtocol()
        protocol._transport = MagicMock()
        request = make_invite()
        addr = ("192.0.2.1", 5060)
        call = protocol.create_call(request, addr)
        assert isinstance(call, IncomingCall)
        assert call.caller == "sip:bob@biloxi.com"

    def test_create_call__custom_class(self):
        """Use the overridden create_call to produce a custom call object."""
        class CustomCall(IncomingCall):
            pass

        class CustomProtocol(IncomingCallProtocol):
            def create_call(self, request, addr) -> CustomCall:
                return CustomCall(request, addr, self._transport)

        protocol = CustomProtocol()
        protocol._transport = MagicMock()
        call = protocol.create_call(make_invite(), ("192.0.2.1", 5060))
        assert isinstance(call, CustomCall)
