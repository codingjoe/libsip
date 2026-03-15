"""Tests for the RTPCall base class and its backward-compatibility shim."""

from unittest.mock import MagicMock

import pytest
from voip.call import Call
from voip.rtp import RealtimeTransportProtocol, RTPCall, RTPPacket
from voip.sip.types import CallerID


def make_call(**kwargs) -> RTPCall:
    """Create an RTPCall with mock rtp/sip for unit testing."""
    defaults: dict = {
        "rtp": MagicMock(spec=RealtimeTransportProtocol),
        "sip": MagicMock(),
    }
    defaults.update(kwargs)
    return RTPCall(**defaults)


class TestCallCompatShim:
    def test_call__is_rtp_call(self):
        """Call is a backward-compatibility alias for RTPCall."""
        assert Call is RTPCall


class TestRTPCall:
    def test_caller__defaults_to_empty_string(self):
        """Caller defaults to an empty CallerID when not provided."""
        call = make_call()
        assert str(call.caller) == ""

    def test_caller__stores_provided_value(self):
        """Caller stores the CallerID passed at construction."""
        call = make_call(caller=CallerID("sip:bob@biloxi.com"))
        assert str(call.caller) == "sip:bob@biloxi.com"

    def test_media__defaults_to_none(self):
        """Media is None when not provided."""
        assert make_call().media is None

    def test_rtp_and_sip_stored_as_fields(self):
        """Rtp and sip back-references are stored on the instance."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_sip = MagicMock()
        call = RTPCall(rtp=mock_rtp, sip=mock_sip)
        assert call.rtp is mock_rtp
        assert call.sip is mock_sip

    def test_datagram_received__drops_malformed_packet(self):
        """Drop datagrams that are too short to be valid RTP silently."""
        make_call().datagram_received(b"data", ("192.0.2.1", 5004))  # must not raise

    def test_datagram_received__dispatches_valid_packet(self):
        """Dispatch a valid RTP datagram to packet_received."""
        import struct  # noqa: PLC0415

        received: list[RTPPacket] = []

        class Capture(RTPCall):
            def packet_received(self, packet: RTPPacket, addr: tuple[str, int]) -> None:
                received.append(packet)

        raw = struct.pack(">BBHII", 0x80, 8, 1, 0, 0) + b"audio"
        Capture(rtp=MagicMock(), sip=MagicMock()).datagram_received(
            raw, ("127.0.0.1", 5004)
        )
        assert len(received) == 1
        assert received[0].payload == b"audio"

    def test_packet_received__noop_by_default(self):
        """packet_received is a no-op in the base class."""
        packet = RTPPacket(
            payload_type=8, sequence_number=1, timestamp=0, ssrc=0, payload=b"x"
        )
        make_call().packet_received(packet, ("192.0.2.1", 5004))  # must not raise

    def test_send_packet__delegates_through_send_datagram(self):
        """send_packet serializes the packet and forwards it via send_datagram."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        call = make_call(rtp=mock_rtp)
        packet = RTPPacket(
            payload_type=8, sequence_number=1, timestamp=0, ssrc=0, payload=b"x"
        )
        call.send_packet(packet, ("192.0.2.1", 5004))
        mock_rtp.send.assert_called_once_with(packet.build(), ("192.0.2.1", 5004))

    def test_send_datagram__delegates_to_rtp(self):
        """send_datagram forwards data through the shared RTP socket."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        call = make_call(rtp=mock_rtp)
        call.send_datagram(b"audio", ("192.0.2.1", 5004))
        mock_rtp.send.assert_called_once_with(b"audio", ("192.0.2.1", 5004))

    def test_send_datagram__encrypts_with_srtp_when_set(self):
        """send_datagram encrypts the payload when an SRTP session is attached."""
        import struct  # noqa: PLC0415

        from voip.srtp import SRTPSession  # noqa: PLC0415

        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        session = SRTPSession.generate()
        call = make_call(rtp=mock_rtp, srtp=session)
        # Use a valid RTP packet so SRTP can compute an auth tag.
        raw = struct.pack(">BBHII", 0x80, 8, 1, 0, 0) + b"audio"
        call.send_datagram(raw, ("192.0.2.1", 5004))
        sent_data = mock_rtp.send.call_args[0][0]
        assert sent_data != raw

    def test_negotiate_codec__raises_not_implemented(self):
        """negotiate_codec raises NotImplementedError in the base class."""
        with pytest.raises(NotImplementedError):
            RTPCall.negotiate_codec(MagicMock())

    async def test_hang_up__raises_not_implemented(self):
        """hang_up raises NotImplementedError in the base class."""
        with pytest.raises(NotImplementedError):
            await make_call().hang_up()
