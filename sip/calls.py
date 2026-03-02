"""SIP call handling."""

from __future__ import annotations

import asyncio

from .aio import SessionInitiationProtocol
from .messages import Request, Response

__all__ = ["IncomingCall", "IncomingCallProtocol", "RTPProtocol"]

_RTP_HEADER_SIZE = 12


class RTPProtocol(asyncio.DatagramProtocol):
    """An asyncio DatagramProtocol that strips RTP headers and dispatches audio payloads."""

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Strip RTP header and forward audio payload."""
        if len(data) > _RTP_HEADER_SIZE:
            self.handle(data[_RTP_HEADER_SIZE:])

    def handle(self, audio: bytes) -> None:
        """Handle incoming audio data. Override in subclasses."""
        return NotImplemented


class IncomingCall(RTPProtocol):
    """An incoming SIP call."""

    def __init__(
        self,
        request: Request,
        addr: tuple[str, int],
        transport: asyncio.DatagramTransport,
    ) -> None:
        self._request = request
        self._addr = addr
        self._transport = transport

    @property
    def caller(self) -> str:
        """Return the caller's SIP address."""
        return self._request.headers.get("From", "")

    async def answer(self) -> None:
        """Answer the call and start receiving audio via RTP."""
        loop = asyncio.get_running_loop()
        rtp_transport, _ = await loop.create_datagram_endpoint(
            lambda: self,
            local_addr=("0.0.0.0", 0),  # noqa: S104
        )
        local_addr = rtp_transport.get_extra_info("sockname")
        sdp = (
            f"v=0\r\n"
            f"c=IN IP4 {local_addr[0]}\r\n"
            f"m=audio {local_addr[1]} RTP/AVP 0\r\n"
        ).encode()
        self._transport.sendto(
            bytes(
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
                        "Content-Length": str(len(sdp)),
                    },
                    body=sdp,
                )
            ),
            self._addr,
        )

    def reject(self, status_code: int = 486, reason: str = "Busy Here") -> None:
        """Reject the call."""
        self._transport.sendto(
            bytes(
                Response(
                    status_code=status_code,
                    reason=reason,
                    headers={
                        key: value
                        for key, value in self._request.headers.items()
                        if key in ("Via", "To", "From", "Call-ID", "CSeq")
                    },
                )
            ),
            self._addr,
        )


class IncomingCallProtocol(SessionInitiationProtocol):
    """SIP protocol with incoming call (INVITE) support."""

    def request_received(self, request: Request, addr: tuple[str, int]) -> None:
        """Dispatch an INVITE request to invite_received."""
        match request.method:
            case "INVITE":
                self.invite_received(self.create_call(request, addr), addr)
            case _:
                return NotImplemented

    def create_call(self, request: Request, addr: tuple[str, int]) -> IncomingCall:
        """Create an IncomingCall for an INVITE. Override to use a custom call class."""
        return IncomingCall(request, addr, self._transport)

    def invite_received(self, call: IncomingCall, addr: tuple[str, int]) -> None:
        """Handle an incoming call. Override in subclasses to process calls."""
        return NotImplemented
