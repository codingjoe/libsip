"""Tests for the SIP INVITE server transaction."""

from __future__ import annotations

import asyncio
import dataclasses
import ipaddress
import logging
from unittest.mock import MagicMock

import pytest
from voip.rtp import RealtimeTransportProtocol, RTPCall
from voip.sdp.messages import SessionDescription
from voip.sdp.types import MediaDescription, RTPPayloadFormat
from voip.sip.messages import Request, Response
from voip.sip.protocol import SessionInitiationProtocol
from voip.sip.transactions import Transaction
from voip.sip.types import SIPStatus

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _FakeTransport:
    """Minimal fake TCP/TLS transport for transaction tests."""

    def __init__(
        self,
        peer: tuple[str, int] = ("192.0.2.1", 5060),
        local: tuple[str, int] = ("127.0.0.1", 5061),
    ):
        self._peer = peer
        self._local = local
        self.written: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.written.append(data)

    def get_extra_info(self, key: str, default=None):
        match key:
            case "sockname":
                return self._local
            case "peername":
                return self._peer
            case "ssl_object":
                return object()  # non-None signals TLS
        return default


class _CapturingSIP(SessionInitiationProtocol):
    """SIP subclass that captures sent responses without real I/O."""

    def __init__(self, peer: tuple[str, int] = ("192.0.2.1", 5060)):
        super().__init__(
            outbound_proxy=("127.0.0.1", 5061),
            aor="sip:test@example.com",
        )
        self._sent: list[Response] = []
        self.transport = _FakeTransport(peer=peer)
        self.local_address = (ipaddress.IPv4Address("127.0.0.1"), 5061)
        self.is_secure = True

    def send(self, message: Response | Request) -> None:
        if isinstance(message, Response):
            self._sent.append(message)


@dataclasses.dataclass
class _MinimalCall(RTPCall):
    """Minimal call subclass that always selects PCMU (payload type 0)."""

    @classmethod
    def negotiate_codec(cls, remote_media: MediaDescription) -> MediaDescription:
        return MediaDescription(
            media="audio",
            port=0,
            proto=remote_media.proto,
            fmt=[RTPPayloadFormat.from_pt(0)],
        )


def _make_invite(
    call_id: str = "test-call-1",
    branch: str = "z9hG4bKtest1",
    sdp: bytes | None = None,
) -> Request:
    headers = {
        "Via": f"SIP/2.0/UDP pc33.atlanta.com;branch={branch}",
        "From": "sip:alice@atlanta.com",
        "To": "sip:bob@biloxi.com",
        "Call-ID": call_id,
        "CSeq": "1 INVITE",
    }
    return Request(
        method="INVITE",
        uri="sip:bob@biloxi.com",
        headers=headers,
        body=SessionDescription.parse(sdp) if sdp else None,
    )


def _make_transaction(
    sip: _CapturingSIP | None = None,
    invite: Request | None = None,
    to_tag: str = "deadbeef12345678",
) -> Transaction:
    sip = sip or _CapturingSIP()
    invite = invite or _make_invite()
    return Transaction(
        branch=invite.via_branch,
        invite=invite,
        to_tag=to_tag,
        sip=sip,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTransactionProperties:
    def test_dialog_headers__extracts_dialog_fields(self):
        """Extract Via, To, From, Call-ID and CSeq from the INVITE."""
        invite = _make_invite()
        transaction = _make_transaction(invite=invite)
        assert transaction.dialog_headers == {
            "Via": invite.headers["Via"],
            "From": invite.headers["From"],
            "To": invite.headers["To"],
            "Call-ID": invite.headers["Call-ID"],
            "CSeq": invite.headers["CSeq"],
        }

    def test_tagged_headers__appends_to_tag(self):
        """Append ``;tag=`` to the To header using the stored to_tag."""
        transaction = _make_transaction(to_tag="deadbeef12345678")
        assert ";tag=deadbeef12345678" in transaction.tagged_headers["To"]

    def test_tagged_headers__empty_to_tag__no_tag_appended(self):
        """Leave the To header unchanged when to_tag is empty."""
        transaction = _make_transaction(to_tag="")
        assert ";tag=" not in transaction.tagged_headers["To"]


class TestTransactionCallReceived:
    def test_call_received__is_noop(self):
        """Default call_received does not raise."""
        transaction = _make_transaction()
        transaction.invite_received(transaction.invite)  # must not raise


class TestTransactionRinging:
    def test_ringing__sends_180(self):
        """Send 180 Ringing with tagged dialog headers via sip.send."""
        sip = _CapturingSIP()
        transaction = _make_transaction(sip)
        transaction.ringing()
        assert len(sip._sent) == 1
        assert sip._sent[0].status_code == SIPStatus.RINGING
        assert ";tag=deadbeef12345678" in sip._sent[0].headers.get("To", "")

    def test_ringing__logs_info(self, caplog):
        """Log a call_ringing event at INFO level."""
        with caplog.at_level(logging.INFO, logger="voip.sip"):
            _make_transaction().ringing()
        assert any("call_ringing" in r.message for r in caplog.records)


class TestTransactionReject:
    def test_reject__sends_486_by_default(self):
        """Send 486 Busy Here when no status code is specified."""
        sip = _CapturingSIP()
        sip._to_tags["test-call-1"] = "deadbeef12345678"
        transaction = _make_transaction(sip)
        transaction.reject()
        assert len(sip._sent) == 1
        assert sip._sent[0].status_code == SIPStatus.BUSY_HERE
        assert ";tag=deadbeef12345678" in sip._sent[0].headers.get("To", "")

    def test_reject__custom_status(self):
        """Send the specified status code and its reason phrase."""
        sip = _CapturingSIP()
        _make_transaction(sip).reject(status_code=SIPStatus.DECLINE)
        assert sip._sent[0].status_code == SIPStatus.DECLINE

    def test_reject__cleans_up_to_tag(self):
        """Remove the call's entry from sip._to_tags after reject."""
        sip = _CapturingSIP()
        sip._to_tags["test-call-1"] = "deadbeef12345678"
        _make_transaction(sip).reject()
        assert "test-call-1" not in sip._to_tags

    def test_reject__logs_info(self, caplog):
        """Log a call_rejected event at INFO level."""
        with caplog.at_level(logging.INFO, logger="voip.sip"):
            _make_transaction().reject()
        assert any("call_rejected" in r.message for r in caplog.records)


class TestTransactionAnswer:
    async def test_answer__rtp_not_ready__logs_error(self, caplog):
        """Log an error and return when _rtp_protocol is None and no init task exists."""
        sip = _CapturingSIP()
        sip.rtp = None
        sip.register_task = None
        with caplog.at_level(logging.ERROR, logger="voip.sip"):
            await _make_transaction(sip).answer(call_class=RTPCall)
        assert "RTP mux not ready" in caplog.text
        assert not sip._sent

    async def test_answer__awaits_init_task_then_fails_if_still_none(self, caplog):
        """Log an error when the init task completes but _rtp_protocol remains None."""
        sip = _CapturingSIP()
        sip.rtp = None
        sip.register_task = asyncio.create_task(asyncio.sleep(0))
        with caplog.at_level(logging.ERROR, logger="voip.sip"):
            await _make_transaction(sip).answer(call_class=RTPCall)
        assert "RTP mux not ready" in caplog.text
        assert not sip._sent

    async def test_answer__sends_200_ok(self):
        """Send 200 OK with SDP when RTP is available."""
        loop = asyncio.get_running_loop()
        sip = _CapturingSIP()
        mux = RealtimeTransportProtocol()
        mux.public_address = loop.create_future()
        mux.public_address.set_result((ipaddress.IPv4Address("127.0.0.1"), 12000))
        sip.rtp = mux
        sip._rtp_transport = MagicMock()
        invite = _make_invite(
            sdp=(
                b"v=0\r\n"
                b"o=- 0 0 IN IP4 192.0.2.1\r\n"
                b"s=-\r\n"
                b"c=IN IP4 192.0.2.1\r\n"
                b"t=0 0\r\n"
                b"m=audio 49170 RTP/AVP 0\r\n"
            )
        )
        await _make_transaction(sip, invite).answer(call_class=_MinimalCall)
        assert sip._sent
        assert sip._sent[0].status_code == 200

    async def test_answer__no_connection__uses_peer_address(self):
        """Fall back to the peer address when no c= line is present in the SDP."""
        loop = asyncio.get_running_loop()
        sip = _CapturingSIP(peer=("10.0.0.1", 5060))
        mux = RealtimeTransportProtocol()
        mux.public_address = loop.create_future()
        mux.public_address.set_result((ipaddress.IPv4Address("127.0.0.1"), 12000))
        sip.rtp = mux
        sip._rtp_transport = MagicMock()
        # SDP without any c= line at session or media level
        invite = _make_invite(
            sdp=(
                b"v=0\r\n"
                b"o=- 0 0 IN IP4 10.0.0.1\r\n"
                b"s=-\r\n"
                b"t=0 0\r\n"
                b"m=audio 49170 RTP/AVP 0\r\n"
            )
        )
        await _make_transaction(sip, invite).answer(call_class=_MinimalCall)
        assert ("10.0.0.1", 49170) in mux.calls


class TestTransactionMakeCall:
    async def test_make_call__raises_not_implemented(self):
        """make_call raises NotImplementedError since UAC is not yet supported."""
        with pytest.raises(NotImplementedError):
            await _make_transaction().make_call(
                "sip:bob@example.com", call_class=RTPCall
            )
