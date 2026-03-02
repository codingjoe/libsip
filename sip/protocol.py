"""SIP asyncio protocol handler."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .messages import Request, Response, parse

if TYPE_CHECKING:
    pass


class SIPProtocol(asyncio.DatagramProtocol):
    """An asyncio protocol handler for the Session Initiation Protocol (RFC 3261)."""

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Dispatch a received datagram to the appropriate handler."""
        match parse(data):
            case Request() as request:
                self.request_received(request, addr)
            case Response() as response:
                self.response_received(response, addr)

    def request_received(self, request: Request, addr: tuple[str, int]) -> None:
        """Handle a received SIP request. Override in subclasses to process requests."""

    def response_received(self, response: Response, addr: tuple[str, int]) -> None:
        """Handle a received SIP response. Override in subclasses to process responses."""
