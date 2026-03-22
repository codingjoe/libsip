"""Tests for the SIP INVITE server transaction."""

from __future__ import annotations

import dataclasses
import ipaddress
import logging

import pytest
from voip.rtp import RealtimeTransportProtocol, RTPCall
from voip.sdp.messages import SessionDescription
from voip.sdp.types import MediaDescription, RTPPayloadFormat
from voip.sip.messages import Request, Response
from voip.sip.protocol import SessionInitiationProtocol
from voip.sip.transactions import Transaction
from voip.sip.types import SIPStatus, SipUri

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

    def __init__(self, **kwargs):
        peer = kwargs.pop("peer", ("192.0.2.1", 5060))
        kwargs.setdefault(
            "transaction_class", getattr(self, "transaction_class", Transaction)
        )
        kwargs.setdefault("rtp", RealtimeTransportProtocol())
        super().__init__(
            outbound_proxy=("127.0.0.1", 5061),
            aor=SipUri.parse("sip:test@example.com"),
            **kwargs,
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
        """Return 180 Ringing with tagged dialog headers."""
        sip = _CapturingSIP()
        transaction = _make_transaction(sip)
        response = transaction.ringing()
        assert response.status_code == SIPStatus.RINGING
        assert ";tag=deadbeef12345678" in response.headers.get("To", "")

    def test_ringing__logs_info(self, caplog):
        """Log a call_ringing event at INFO level."""
        with caplog.at_level(logging.INFO, logger="voip.sip"):
            _make_transaction().ringing()
        assert any("call_ringing" in r.message for r in caplog.records)


class TestTransactionReject:
    def test_reject__sends_486_by_default(self):
        """Return 486 Busy Here when no status code is specified."""
        sip = _CapturingSIP()
        sip._to_tags["test-call-1"] = "deadbeef12345678"
        transaction = _make_transaction(sip)
        response = transaction.reject()
        assert response.status_code == SIPStatus.BUSY_HERE
        assert ";tag=deadbeef12345678" in response.headers.get("To", "")

    def test_reject__custom_status(self):
        """Return the specified status code and its reason phrase."""
        sip = _CapturingSIP()
        response = _make_transaction(sip).reject(status_code=SIPStatus.DECLINE)
        assert response.status_code == SIPStatus.DECLINE

    def test_reject__returns_tagged_headers(self):
        """Include To tag in the returned response headers."""
        transaction = _make_transaction(to_tag="test-tag")
        response = transaction.reject()
        assert ";tag=test-tag" in response.headers.get("To", "")

    def test_reject__logs_info(self, caplog):
        """Log a call_rejected event at INFO level."""
        with caplog.at_level(logging.INFO, logger="voip.sip"):
            _make_transaction().reject()
        assert any("call_rejected" in r.message for r in caplog.records)


class TestTransactionAnswer:
    def test_answer__returns_200_ok(self):
        """Return 200 OK with SDP."""
        sip = _CapturingSIP()
        mux = RealtimeTransportProtocol()
        mux.public_address = (ipaddress.IPv4Address("127.0.0.1"), 12000)
        sip.rtp = mux
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
        response = _make_transaction(sip, invite).answer(call_class=_MinimalCall)
        assert response.status_code == 200
        assert response.phrase == "OK"
        assert response.body is not None

    def test_answer__negotiates_codec(self):
        """Call negotiate_codec on the call class to select codec."""
        sip = _CapturingSIP()
        mux = RealtimeTransportProtocol()
        mux.public_address = (ipaddress.IPv4Address("127.0.0.1"), 12000)
        sip.rtp = mux
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
        response = _make_transaction(sip, invite).answer(call_class=_MinimalCall)
        assert response.status_code == 200

    def test_answer__no_sdp__creates_default_media(self):
        """Create default audio media when INVITE has no SDP."""
        sip = _CapturingSIP()
        mux = RealtimeTransportProtocol()
        mux.public_address = (ipaddress.IPv4Address("127.0.0.1"), 12000)
        sip.rtp = mux
        invite = _make_invite(sdp=None)
        response = _make_transaction(sip, invite).answer(call_class=_MinimalCall)
        assert response.status_code == 200
        assert response.body is not None


class TestTransactionMakeCall:
    async def test_make_call__raises_not_implemented(self):
        """make_call raises NotImplementedError since UAC is not yet supported."""
        with pytest.raises(NotImplementedError):
            await _make_transaction().make_call(
                "sip:bob@example.com", call_class=RTPCall
            )
