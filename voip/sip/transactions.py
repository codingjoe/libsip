"""SIP INVITE server transaction (RFC 3261 §17.2)."""

from __future__ import annotations

import dataclasses
import ipaddress
import json
import logging
import secrets
import typing

from voip.rtp import RTPCall
from voip.sdp.messages import SessionDescription
from voip.sdp.types import (
    Attribute,
    ConnectionData,
    MediaDescription,
    Origin,
    RTPPayloadFormat,
    Timing,
)
from voip.srtp import SRTPSession

from .messages import Request, Response
from .types import CallerID, SIPStatus

if typing.TYPE_CHECKING:
    from .protocol import SessionInitiationProtocol

logger = logging.getLogger("voip.sip")

__all__ = [
    "Transaction",
]


@dataclasses.dataclass(kw_only=True, slots=True)
class Transaction:
    """SIP INVITE server transaction [RFC 3261 §17.2].

    Encapsulates the state and behavior of a single INVITE dialog.
    The SIP layer creates one instance per incoming INVITE, keyed by
    Via branch (RFC 3261 §17.1.3).

    Override `call_received` in a subclass to react to the call without
    subclassing
    [`SessionInitiationProtocol`][voip.sip.protocol.SessionInitiationProtocol]:

    ```python
    class MyTransaction(Transaction):
        def call_received(self) -> None:
            asyncio.create_task(self.answer(call_class=MyCall))
    ```

    Register the subclass on the session:

    ```python
    class MySession(SessionInitiationProtocol):
        transaction_class = MyTransaction
    ```

    [RFC 3261 §17.2]: https://datatracker.ietf.org/doc/html/rfc3261#section-17.2

    Args:
        branch: Via branch token that uniquely identifies this transaction.
        invite: The SIP INVITE request that started this transaction.
        to_tag: Locally generated To tag for the dialog [RFC 3261 §8.2.6.2].
        sip: The owning
            [`SessionInitiationProtocol`][voip.sip.protocol.SessionInitiationProtocol]
            instance used to send responses.
    """

    branch: str
    invite: Request
    to_tag: str
    sip: SessionInitiationProtocol = dataclasses.field(repr=False)

    @property
    def dialog_headers(self) -> dict[str, str]:
        """Dialog headers extracted from the INVITE.

        Returns:
            A dict containing Via, To, From, Call-ID, and CSeq headers.
        """
        return {
            key: value
            for key, value in self.invite.headers.items()
            if key in ("Via", "To", "From", "Call-ID", "CSeq")
        }

    @property
    def tagged_headers(self) -> dict[str, str]:
        """Dialog headers with the locally generated To tag appended.

        Per [RFC 3261 §8.2.6.2], UAS responses to an INVITE must include a To
        tag that identifies the dialog from the server side.

        [RFC 3261 §8.2.6.2]: https://datatracker.ietf.org/doc/html/rfc3261#section-8.2.6.2

        Returns:
            A dict of dialog headers where the To header carries the `to_tag`.
        """
        headers = self.dialog_headers
        return {
            **headers,
            "To": headers.get("To", "")
            + (f";tag={self.to_tag}" if self.to_tag else ""),
        }

    def call_received(self) -> None:
        """Handle the incoming call.

        Override in subclasses to decide whether to answer, ring, or reject.
        The base implementation is a no-op.
        """

    def ringing(self) -> None:
        """Send a 180 Ringing provisional response [RFC 3261 §21.1.2].

        Call before `answer` to notify the caller that the UA is alerting the
        user.

        [RFC 3261 §21.1.2]: https://datatracker.ietf.org/doc/html/rfc3261#section-21.1.2
        """
        call_id = self.invite.headers.get("Call-ID", "")
        caller = CallerID(self.invite.headers.get("From", ""))
        logger.info(
            json.dumps(
                {"event": "call_ringing", "caller": repr(caller), "call_id": call_id}
            ),
            extra={"caller": repr(caller), "call_id": call_id},
        )
        self.sip.send(
            Response(
                status_code=SIPStatus.RINGING,
                phrase=SIPStatus.RINGING.phrase,
                headers=self.tagged_headers,
            )
        )

    def reject(self, status_code: SIPStatus = SIPStatus.BUSY_HERE) -> None:
        """Reject the incoming call.

        Args:
            status_code: SIP response status code (default: 486 Busy Here).
        """
        call_id = self.invite.headers.get("Call-ID", "")
        peer = (
            self.sip.transport.get_extra_info("peername")
            if self.sip.transport
            else None
        )
        caller = CallerID(self.invite.headers.get("From", ""))
        logger.info(
            json.dumps(
                {
                    "event": "call_rejected",
                    "caller": repr(caller),
                    "ip": peer[0] if peer else None,
                    "call_id": call_id,
                    "status": status_code,
                    "reason": status_code.phrase,
                }
            ),
            extra={
                "caller": repr(caller),
                "ip": peer[0] if peer else None,
                "call_id": call_id,
                "status": status_code,
            },
        )
        self.sip.send(
            Response(
                status_code=status_code,
                phrase=status_code.phrase,
                headers=self.tagged_headers,
            )
        )
        self.sip._to_tags.pop(call_id, None)

    async def answer(
        self, *, call_class: type[RTPCall], **call_kwargs: typing.Any
    ) -> None:
        """Answer the call by setting up RTP and sending 200 OK with SDP.

        Example:
            Call from within `call_received`:

            ```python
            def call_received(self) -> None:
                asyncio.create_task(self.answer(call_class=MyCall))
            ```

        Args:
            call_class: [`RTPCall`][voip.rtp.RTPCall] subclass whose
                `negotiate_codec` selects the codec.  The class is
                constructed with `rtp`, `sip`, `caller`, `media`, and
                `srtp` keyword arguments.
            **call_kwargs: Additional keyword arguments forwarded to the
                call class constructor.

        Raises:
            NotImplementedError: When `negotiate_codec` raises (no supported
                codec in the remote SDP offer).
        """
        call_id = self.invite.headers.get("Call-ID", "")
        if self.sip._rtp_protocol is None:
            if self.sip._initialize_task is not None:
                await self.sip._initialize_task
            if self.sip._rtp_protocol is None:
                logger.error("RTP mux not ready; cannot answer call")
                return
        peer = (
            self.sip.transport.get_extra_info("peername")
            if self.sip.transport
            else None
        )
        caller = CallerID(self.invite.headers.get("From", ""))
        logger.info(
            json.dumps(
                {
                    "event": "call_answered",
                    "caller": repr(caller),
                    "ip": peer[0] if peer else None,
                    "call_id": call_id,
                }
            ),
            extra={
                "caller": repr(caller),
                "ip": peer[0] if peer else None,
                "call_id": call_id,
            },
        )
        remote_audio = next(
            (
                m
                for m in (self.invite.body.media if self.invite.body else [])
                if m.media == "audio"
            ),
            None,
        )
        if remote_audio is not None:
            negotiated_media = call_class.negotiate_codec(remote_audio)
        else:
            negotiated_media = MediaDescription(
                media="audio",
                port=0,
                proto="RTP/SAVP",
                fmt=[RTPPayloadFormat.from_pt(0)],
            )

        use_srtp = negotiated_media.proto == "RTP/SAVP"
        srtp_session = SRTPSession.generate() if use_srtp else None

        call_handler = call_class(
            rtp=self.sip._rtp_protocol,
            sip=self.sip,
            caller=caller,
            media=negotiated_media,
            srtp=srtp_session,
            **call_kwargs,
        )
        if remote_audio is not None and remote_audio.port != 0:
            media_connection = remote_audio.connection
            session_connection = (
                self.invite.body.connection if self.invite.body else None
            )
            connection = media_connection or session_connection
            if connection is not None:
                remote_ip = connection.connection_address
            else:
                remote_ip = peer[0] if peer else "0.0.0.0"  # noqa: S104
            remote_rtp_address: tuple[str, int] | None = (remote_ip, remote_audio.port)
        else:
            remote_rtp_address = None
        self.sip._rtp_protocol.register_call(remote_rtp_address, call_handler)
        self.sip._call_rtp_addrs[call_id] = remote_rtp_address

        if remote_rtp_address is not None:
            self.sip._rtp_protocol.send(b"\x00", remote_rtp_address)

        record_route = self.invite.headers.get("Record-Route")
        session_id = str(secrets.randbelow(2**32) + 1)
        rtp_public = await self.sip._rtp_protocol.public_address
        sdp_media_attributes = [Attribute(name="sendrecv")]
        if srtp_session is not None:
            sdp_media_attributes.append(
                Attribute(name="crypto", value=srtp_session.sdes_attribute)
            )
        self.sip.send(
            Response(
                status_code=SIPStatus.OK,
                phrase=SIPStatus.OK.phrase,
                headers={
                    **self.tagged_headers,
                    **({"Record-Route": record_route} if record_route else {}),
                    "Contact": self.sip._build_contact(),
                    "Allow": self.sip.allow_header,
                    "Supported": "replaces",
                    "Content-Type": "application/sdp",
                },
                body=SessionDescription(
                    origin=Origin(
                        username="-",
                        sess_id=session_id,
                        sess_version=session_id,
                        nettype="IN",
                        addrtype="IP6"
                        if isinstance(rtp_public[0], ipaddress.IPv6Address)
                        else "IP4",
                        unicast_address=str(rtp_public[0]),
                    ),
                    timings=[Timing(start_time=0, stop_time=0)],
                    connection=ConnectionData(
                        nettype="IN",
                        addrtype="IP6"
                        if isinstance(rtp_public[0], ipaddress.IPv6Address)
                        else "IP4",
                        connection_address=str(rtp_public[0]),
                    ),
                    media=[
                        MediaDescription(
                            media="audio",
                            port=rtp_public[1],
                            proto=negotiated_media.proto,
                            fmt=negotiated_media.fmt,
                            attributes=sdp_media_attributes,
                        )
                    ],
                ),
            )
        )
        self.sip._to_tags.pop(call_id, None)

    async def make_call(
        self,
        target: str,
        *,
        call_class: type[RTPCall],
        **call_kwargs: typing.Any,
    ) -> None:
        """Initiate an outgoing call to `target`.

        Args:
            target: SIP URI of the callee (e.g. ``"sip:bob@example.com"``).
            call_class: [`RTPCall`][voip.rtp.RTPCall] subclass for the
                outbound call leg.
            **call_kwargs: Additional keyword arguments forwarded to the
                call class constructor.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("make_call is not yet implemented")
