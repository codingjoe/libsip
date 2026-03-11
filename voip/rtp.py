"""
Real-time Transport Protocol (RTP) implementation of RFC 3550.

See also: https://datatracker.ietf.org/doc/html/rfc3550#section-5
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import json
import logging
import struct
import uuid
from typing import ClassVar

from voip.sdp.types import MediaDescription, RTPPayloadFormat
from voip.sip.types import CallerID
from voip.stun import MAGIC_COOKIE, STUNMessageType, _parse_stun_response

__all__ = ["RTP", "RTPPacket", "RTPPayloadType", "RealtimeTransportProtocol"]

logger = logging.getLogger(__name__)


class RTPPayloadType(enum.IntEnum):
    """Common RTP payload types, aligned with SDP media format identifiers.

    Static payload types (0–95) are defined by RFC 3551.
    Dynamic payload types (96–127) are negotiated via SDP.
    Opus uses payload type 111 per RFC 7587.
    """

    PCMU = 0  # G.711 µ-law (RFC 3551)
    PCMA = 8  # G.711 A-law (RFC 3551)
    G722 = 9  # G.722 (RFC 3551)
    OPUS = 111  # RFC 7587 (dynamic)


@dataclasses.dataclass
class RTPPacket:
    """A parsed RTP packet (RFC 3550 §5.1)."""

    payload_type: int
    sequence_number: int
    timestamp: int
    ssrc: int
    payload: bytes

    #: Fixed RTP header size in bytes (RFC 3550 §5.1).
    header_size: int = dataclasses.field(default=12, init=False, repr=False)

    @classmethod
    def parse(cls, data: bytes) -> RTPPacket:
        """Parse raw RTP bytes into an RTPPacket."""
        if len(data) < 12:
            raise ValueError(f"RTP packet too short: {len(data)} bytes")
        payload_type = data[1] & 0x7F
        sequence_number = (data[2] << 8) | data[3]
        timestamp = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
        ssrc = (data[8] << 24) | (data[9] << 16) | (data[10] << 8) | data[11]
        return cls(
            payload_type=payload_type,
            sequence_number=sequence_number,
            timestamp=timestamp,
            ssrc=ssrc,
            payload=data[12:],
        )


class RealtimeTransportProtocol(asyncio.DatagramProtocol):
    """Base class for RTP audio call handlers (RFC 3550).

    Subclass and override :meth:`audio_received` to process incoming audio.
    Override :meth:`negotiate_codec` to customise codec selection.

    Set :attr:`chunk_duration` to buffer multiple packets before each
    :meth:`audio_received` call.  The default of ``0`` passes each packet
    individually as a single-element list.
    """

    #: Fixed RTP header size in bytes (RFC 3550 §5.1).
    rtp_header_size: int = 12

    #: Seconds of audio to buffer before emitting an :meth:`audio_received` event.
    #: ``0`` (default) emits one event per RTP packet.
    chunk_duration: ClassVar[int] = 0

    #: Preferred codecs, highest to lowest priority.
    PREFERRED_CODECS: ClassVar[list[RTPPayloadFormat]] = [
        RTPPayloadFormat(
            payload_type=RTPPayloadType.OPUS,
            encoding_name="opus",
            sample_rate=48000,
            channels=2,
        ),
        RTPPayloadFormat(payload_type=RTPPayloadType.G722),
        RTPPayloadFormat(payload_type=RTPPayloadType.PCMA),
        RTPPayloadFormat(payload_type=RTPPayloadType.PCMU),
    ]

    def __init__(self, caller: str = "", media: MediaDescription | None = None) -> None:
        super().__init__()
        self.caller = caller
        self.media = media
        if media is not None and media.fmt:
            fmt = media.fmt[0]
            self.payload_type: int = fmt.payload_type
            self.sample_rate: int = fmt.sample_rate or 8000
            caller_id = CallerID(self.caller)
            logger.info(
                json.dumps(
                    {
                        "event": "call_started",
                        "caller": repr(caller_id),
                        "codec": fmt.encoding_name or "unknown",
                        "sample_rate": fmt.sample_rate or 0,
                        "channels": fmt.channels,
                        "payload_type": fmt.payload_type,
                    }
                ),
                extra={
                    "caller": repr(caller_id),
                    "codec": fmt.encoding_name,
                    "payload_type": fmt.payload_type,
                },
            )
            frame_size = fmt.frame_size
        else:
            self.payload_type: int = 0
            self.sample_rate: int = 8000
            frame_size = 160

        self._audio_buffer: list[bytes] = []
        self._packet_threshold: int = (
            self.sample_rate * self.chunk_duration // frame_size
            if self.chunk_duration
            else 1
        )
        self._stun_pending: dict[bytes, asyncio.Future[tuple[str, int]]] = {}

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:  # type: ignore[override]
        """Store the transport so that :meth:`stun_discover` can use the real socket."""
        self._transport = transport

    @classmethod
    def negotiate_codec(cls, remote_media: MediaDescription) -> MediaDescription:
        """Select the best codec from the offered SDP and return a negotiated MediaDescription.

        Iterates :attr:`PREFERRED_CODECS` in priority order, matching by payload
        type or encoding name.  Raises :exc:`NotImplementedError` when no codec
        matches.
        """
        if not remote_media.fmt:
            raise NotImplementedError("Remote SDP offer contains no audio formats")

        remote_fmts = {f.payload_type for f in remote_media.fmt}
        for preferred in cls.PREFERRED_CODECS:
            # Match by payload type number.
            if preferred.payload_type in remote_fmts:
                remote_fmt = remote_media.get_format(preferred.payload_type)
                codec = (
                    remote_fmt if remote_fmt and remote_fmt.encoding_name else preferred
                )
                return MediaDescription(
                    media="audio",
                    port=0,
                    proto="RTP/AVP",
                    fmt=[codec],
                )
            # Match by encoding name for dynamic payload types.
            for remote_fmt in remote_media.fmt:
                if (
                    remote_fmt.encoding_name is not None
                    and remote_fmt.encoding_name.lower()
                    == preferred.encoding_name.lower()
                ):
                    return MediaDescription(
                        media="audio",
                        port=0,
                        proto="RTP/AVP",
                        fmt=[remote_fmt],
                    )

        raise NotImplementedError(
            f"No supported codec found in remote offer "
            f"{[f.payload_type for f in remote_media.fmt]!r}. "
            f"Supported: {[c.encoding_name for c in cls.PREFERRED_CODECS]!r}"
        )

    def datagram_received(self, data: bytes, address: tuple[str, int]) -> None:
        """Multiplex STUN responses (RFC 7983) and RTP audio packets."""
        # RFC 7983: first byte in [0, 3] indicates a STUN packet.
        if data and data[0] < 4:
            if len(data) >= 20:
                tid = data[8:20]
                fut = self._stun_pending.get(tid)
                if fut is not None:
                    _parse_stun_response(data, tid, fut)
            return
        try:
            packet = RTPPacket.parse(data)
        except ValueError:
            return
        if not packet.payload:
            return
        self._audio_buffer.append(packet.payload)
        while len(self._audio_buffer) >= self._packet_threshold:
            packets = self._audio_buffer[: self._packet_threshold]
            self._audio_buffer = self._audio_buffer[self._packet_threshold :]
            self.audio_received(packets)

    async def stun_discover(
        self, host: str, port: int = 3478, timeout_secs: float = 3.0
    ) -> tuple[str, int]:
        """Discover the public IP:port for this RTP socket via STUN (RFC 5389).

        Unlike the module-level :func:`~voip.stun.stun_discover`, this method
        sends the STUN Binding Request through the actual RTP transport socket so
        the server observes the same NAT mapping used by real-time audio traffic.
        STUN responses are demultiplexed from incoming RTP traffic using the
        first-byte heuristic defined in RFC 7983.

        Args:
            host: STUN server hostname or IP address.
            port: STUN server UDP port (default 3478).
            timeout_secs: Seconds to wait for a STUN response.

        Returns:
            A ``(public_ip, public_port)`` tuple as seen by the STUN server.

        Raises:
            asyncio.TimeoutError: If the server does not respond in time.
            RuntimeError: If the STUN response contains no address attribute.
        """
        loop = asyncio.get_running_loop()
        transaction_id = uuid.uuid4().bytes[:12]
        future: asyncio.Future[tuple[str, int]] = loop.create_future()
        self._stun_pending[transaction_id] = future
        request = struct.pack(
            ">HHI12s",
            STUNMessageType.BINDING_REQUEST,
            0,
            MAGIC_COOKIE,
            transaction_id,
        )
        logger.debug("Sending STUN Binding Request to %s:%s via RTP socket", host, port)
        self._transport.sendto(request, (host, port))
        try:
            return await asyncio.wait_for(future, timeout_secs)
        finally:
            self._stun_pending.pop(transaction_id, None)

    def audio_received(self, packets: list[bytes]) -> None:
        """Handle a buffered audio frame. Override in subclasses.

        Called with a list of :attr:`~RealtimeTransportProtocol._packet_threshold`
        raw RTP payloads representing one audio chunk.
        """


#: Short alias for :class:`RealtimeTransportProtocol`.
RTP = RealtimeTransportProtocol
