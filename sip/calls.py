"""SIP call handling."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from .aio import SessionInitiationProtocol
from .messages import Message, Request, Response

__all__ = ["IncomingCall", "IncomingCallProtocol", "RTPProtocol"]


class RTPProtocol(asyncio.DatagramProtocol):
    """An asyncio DatagramProtocol for receiving RTP audio streams (RFC 3550)."""

    rtp_header_size = 12

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Strip the fixed-size RTP header and forward the audio payload."""
        if len(data) > self.rtp_header_size:
            self.audio_received(data[self.rtp_header_size:])

    def audio_received(self, data: bytes) -> None:
        """Handle a decoded RTP audio payload. Override in subclasses."""
        return NotImplemented


class IncomingCall(RTPProtocol):
    """An inbound SIP call: answers or rejects the INVITE and receives Opus audio via RTP."""

    def __init__(
        self,
        request: Request,
        addr: tuple[str, int],
        send: Callable[..., None],
    ) -> None:
        self._request = request
        self._addr = addr
        self._send = send

    @property
    def caller(self) -> str:
        """Return the caller's SIP address."""
        return self._request.headers.get("From", "")

    async def answer(self) -> None:
        """Answer the call and start receiving Opus audio via RTP (RFC 7587)."""
        loop = asyncio.get_running_loop()
        rtp_transport, _ = await loop.create_datagram_endpoint(
            lambda: self,
            local_addr=("0.0.0.0", 0),  # noqa: S104
        )
        local_addr = rtp_transport.get_extra_info("sockname")
        sdp = (
            f"v=0\r\n"
            f"c=IN IP4 {local_addr[0]}\r\n"
            f"m=audio {local_addr[1]} RTP/AVP 111\r\n"
            f"a=rtpmap:111 opus/48000/2\r\n"
        ).encode()
        self._send(
            Response(
                status_code=200,
                reason="OK",
                headers={
                    **{
                        key: value
                        for key, value in self._request.headers.items()
                        if key in ("Via", "To", "From", "Call-ID", "CSeq")
                    },
                    "Content-Type": "application/sdp",
                },
                body=sdp,
            ),
            self._addr,
        )

    def reject(self, status_code: int = 486, reason: str = "Busy Here") -> None:
        """Reject the call."""
        self._send(
            Response(
                status_code=status_code,
                reason=reason,
                headers={
                    key: value
                    for key, value in self._request.headers.items()
                    if key in ("Via", "To", "From", "Call-ID", "CSeq")
                },
            ),
            self._addr,
        )


class IncomingCallProtocol(SessionInitiationProtocol):
    """SIP protocol with incoming call (INVITE) support."""

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        """Store the transport for sending SIP messages."""
        self._transport = transport

    def send(self, message: Message, addr: tuple[str, int]) -> None:
        """Serialize and send a SIP message to the given address."""
        self._transport.sendto(bytes(message), addr)

    def request_received(self, request: Request, addr: tuple[str, int]) -> None:
        """Dispatch an INVITE request to invite_received."""
        match request.method:
            case "INVITE":
                self.invite_received(self.create_call(request, addr), addr)
            case _:
                return NotImplemented

    def create_call(self, request: Request, addr: tuple[str, int]) -> IncomingCall:
        """Create an IncomingCall for an INVITE. Override to use a custom call class."""
        return IncomingCall(request, addr, self.send)

    def invite_received(self, call: IncomingCall, addr: tuple[str, int]) -> None:
        """Handle an incoming call. Override in subclasses to process calls."""
        return NotImplemented
