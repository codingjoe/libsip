"""
SIP asyncio protocol handler.

See also: https://datatracker.ietf.org/doc/html/rfc3261
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import hashlib
import ipaddress
import json
import logging
import re
import secrets
import socket
import typing
import uuid

from voip.rtp import RealtimeTransportProtocol

from . import types
from .exceptions import RegistrationError
from .messages import Message, Request, Response
from .transactions import Transaction
from .types import (
    CallerID,
    DigestAlgorithm,
    DigestQoP,
    SIPMethod,
    SIPStatus,
    _format_host,
)

logger = logging.getLogger("voip.sip")

__all__ = ["SIP", "SessionInitiationProtocol", "Transaction"]


@dataclasses.dataclass(kw_only=True, slots=True)
class SessionInitiationProtocol(asyncio.Protocol):
    """
    SIP User Agent Client (UAC) over TLS/TCP [RFC 3261][RFC 3261].

    Handles incoming calls and, optionally, carrier registration with digest
    authentication [RFC 3261 §22].  All signaling is sent over a single
    persistent TLS/TCP connection.

    ```python
    class MyTransaction(Transaction):
        def call_received(self, request: Request) -> None:
            asyncio.create_task(self.answer(call_class=MyCall))

    class MySession(SessionInitiationProtocol):
        transaction_class = MyTransaction
    ```

    To register with a carrier on startup, pass the registration parameters:

    ```python
    session = SessionInitiationProtocol(
        aor="sips:alice@example.com",
        username="alice",
        password="secret",
        # Optional: connect via a separate outbound proxy
        # outbound_proxy=("proxy.carrier.com", 5061),
    )
    ```

    [RFC 3261]: https://datatracker.ietf.org/doc/html/rfc3261
    [RFC 3261 §22]: https://datatracker.ietf.org/doc/html/rfc3261#section-22

    Attributes:
        VIA_BRANCH_PREFIX:
            RFC 3261 §8.1.1.7 Via branch magic cookie (indicates RFC 3261 compliance).

    Args:
        keepalive_interval: Keep-alive ping interval. Should be between 30 and 90 seconds.

    """

    #: RFC 3261 §8.1.1.7 Via branch magic cookie (indicates RFC 3261 compliance).
    VIA_BRANCH_PREFIX: typing.ClassVar[str] = "z9hG4bK"
    #: Transaction class used to handle incoming INVITE dialogs.
    #: Override in subclasses to inject a custom `Transaction` subclass.
    transaction_class: type[Transaction]

    _transactions: dict[str, Transaction] = dataclasses.field(
        init=False, default_factory=dict
    )
    _to_tags: dict[str, str] = dataclasses.field(init=False, default_factory=dict)
    rtp: RealtimeTransportProtocol
    _keepalive_task: asyncio.Task | None = dataclasses.field(init=False, default=None)
    _call_rtp_addrs: dict[str, tuple[str, int] | None] = dataclasses.field(
        init=False, default_factory=dict
    )
    _buffer: bytearray = dataclasses.field(init=False, default_factory=bytearray)
    disconnected_event: asyncio.Event = dataclasses.field(
        init=False, default_factory=asyncio.Event
    )
    #: RFC 3261 §8.1.2 — outbound SIP proxy address ``(host, port)``.
    #: When ``None`` the caller connects directly to the registrar server.
    #: The address may differ from the registrar domain derived from
    #: `aor` (e.g. ``proxy.carrier.com`` vs ``carrier.com``).
    aor: types.SipUri
    outbound_proxy: (
        tuple[ipaddress.IPv4Address | ipaddress.IPv6Address | str, int] | None
    ) = None
    rtp = RealtimeTransportProtocol
    keepalive_interval: datetime.timedelta = datetime.timedelta(seconds=30)
    call_id: str = dataclasses.field(init=False)
    cseq: int = dataclasses.field(init=False, default=0)
    #: Local TCP socket address (host, port) — set when connection is established.
    local_address: tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, int] = (
        dataclasses.field(init=False)
    )
    transport: asyncio.Transport | None = dataclasses.field(init=False, default=None)
    #: True when the underlying transport is TLS-wrapped; False for plain TCP.
    is_secure: bool = dataclasses.field(init=False, default=False)

    def __post_init__(self):
        self.call_id = f"{uuid.uuid4()}@{socket.gethostname()}"

    def connection_made(self, transport: asyncio.Transport) -> None:  # type: ignore[override]
        """Store the TLS/TCP transport and start RTP mux + carrier registration."""
        self.transport = transport
        # IPv6 sockets return a 4-tuple (host, port, flowinfo, scope_id);
        # we only need the first two elements.
        sockname = transport.get_extra_info("sockname")
        print(sockname)
        host, port = sockname[0], sockname[1]
        self.local_address = (ipaddress.ip_address(host), port)
        self.is_secure = transport.get_extra_info("ssl_object") is not None
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.register())
            self._keepalive_task = loop.create_task(self.send_keepalive())
        except RuntimeError:
            pass  # no running loop in synchronous test setups

    async def send_keepalive(self) -> None:
        while True:
            await asyncio.sleep(self.keepalive_interval.total_seconds())
            if self.transport is None:
                return
            logger.debug("Sending RFC 5626 §4.4.1 keep-alive ping")
            self.transport.write(b"\r\n\r\n")

    def data_received(self, data: bytes) -> None:
        self._buffer.extend(data)
        while True:
            end_of_headers = self._buffer.find(b"\r\n\r\n")
            if end_of_headers == -1:
                break
            header_bytes = bytes(self._buffer[:end_of_headers])
            # Determine body length from Content-Length header.
            content_length = 0
            for line in header_bytes.decode(errors="replace").split("\r\n")[1:]:
                low = line.lower()
                if low.startswith("content-length:"):
                    try:
                        content_length = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                    break
            message_end = end_of_headers + 4 + content_length
            if len(self._buffer) < message_end:
                break
            message_data = bytes(self._buffer[:message_end])
            del self._buffer[:message_end]
            addr = self.transport.get_extra_info("peername") if self.transport else None
            self.packet_received(message_data, addr)

    def packet_received(self, data: bytes, addr: tuple[str, int] | None) -> None:
        """Handle RFC 5626 keepalive pings, then dispatch SIP messages."""
        if data == b"\r\n\r\n":  # RFC 5626 §4.4.1 double-CRLF keepalive ping
            logger.debug("RFC 5626 keepalive from %s, sending pong", addr)
            if self.transport:
                self.transport.write(b"\r\n")
            return
        match Message.parse(data):
            case Request() as request:
                self.request_received(request, addr)
            case Response() as response:
                self.response_received(response, addr)

    def send(self, message: Response | Request) -> None:
        """Serialize and send a SIP message over the TLS/TCP connection."""
        logger.debug("Sending %r", message)
        if self.transport is not None:
            self.transport.write(bytes(message))

    def close(self) -> None:
        """Close the TLS/TCP transport and the RTP mux."""
        if self.transport is not None:
            self.transport.close()

    def _cleanup_rtp_call(self, call_id: str) -> None:
        """Remove the call handler registered with the shared RTP mux, if any."""
        if call_id in self._call_rtp_addrs and self.rtp is not None:
            self.rtp.unregister_call(self._call_rtp_addrs.pop(call_id))

    @property
    def allowed_methods(self) -> frozenset[SIPMethod]:
        """SIP methods supported by this UA.

        A method is included when the class defines a ``<method_lower>_received``
        handler (e.g. ``register_received`` enables REGISTER).

        Returns:
            Frozenset of [`SIPMethod`][voip.sip.types.SIPMethod] values.
        """
        return frozenset(
            m
            for m in SIPMethod
            if hasattr(self.transaction_class, f"{m.lower()}_received")
        )

    @property
    def allow_header(self) -> str:
        """Comma-separated Allow header value in SIPMethod enum order."""
        return ", ".join(m for m in SIPMethod if m in self.allowed_methods)

    def method_not_allowed(self, request: Request) -> None:
        """Respond with 405 Method Not Allowed.

        Override to customise the error response or add logging.

        Args:
            request: The unhandled SIP request.
        """
        logger.warning("SIP method %r is not supported", request.method)
        dialog_headers = {
            key: value
            for key, value in request.headers.items()
            if key in ("Via", "To", "From", "Call-ID", "CSeq")
        }
        self.send(
            Response(
                status_code=SIPStatus.METHOD_NOT_ALLOWED,
                phrase=SIPStatus.METHOD_NOT_ALLOWED.phrase,
                headers={**dialog_headers, "Allow": self.allow_header},
            ),
        )

    def request_received(self, request: Request, addr: tuple[str, int]) -> None:
        """Dispatch an inbound SIP request to the appropriate handler.

        Args:
            request: The parsed SIP request.
            addr: The remote address the request arrived from.
        """
        if request.method == SIPMethod.INVITE:
            trying = Response(
                status_code=SIPStatus.TRYING,
                phrase=SIPStatus.TRYING.phrase,
                headers={
                    key: value
                    for key, value in request.headers.items()
                    if key in ("Via", "To", "From", "Call-ID", "CSeq")
                },
            )
            self.send(trying)
        try:
            tx = self._transactions[request.via_branch]
            if request.method == SIPMethod.INVITE:
                return  # INVITE retransmission handled by transaction layer
        except KeyError:
            call_id = request.headers.get("Call-ID", "")
            caller = CallerID(request.headers.get("From", ""))
            try:
                to_tag = self._to_tags[call_id]
            except KeyError:
                to_tag = secrets.token_hex(8)
                self._to_tags[call_id] = to_tag
            tx = self.transaction_class(
                branch=request.via_branch,
                invite=request,
                to_tag=to_tag,
                sip=self,
            )
            logger.info(
                json.dumps(
                    {
                        "event": "incoming_call",
                        "caller": repr(caller),
                        "call_id": call_id,
                    }
                ),
                extra={"caller": repr(caller), "call_id": call_id},
            )
            self._transactions[request.via_branch] = tx
        handler: typing.Callable[[Request], Response | None] = getattr(
            tx, f"{request.method.lower()}_received", self.method_not_allowed
        )
        if response := handler(request):
            self.send(response)
        if request.method == SIPMethod.CANCEL:
            # RFC 3261 §9.2: CANCEL also triggers a 487 for the INVITE.
            # We look up the transaction by the CANCEL's branch (which is the same).
            if tx and tx.invite.method == SIPMethod.INVITE:
                terminated = tx.reject(status_code=SIPStatus.REQUEST_TERMINATED)
                self.send(terminated)
        if request.method == SIPMethod.BYE:
            self._cleanup_rtp_call(request.headers.get("Call-ID", ""))
        if request.method in (SIPMethod.ACK, SIPMethod.CANCEL, SIPMethod.BYE):
            self._transactions.pop(request.via_branch, None)
            self._to_tags.pop(request.headers.get("Call-ID", ""), None)

    def response_received(
        self, response: Response, addr: tuple[str, int] | None
    ) -> None:
        """Handle REGISTER responses including digest auth challenges (RFC 3261 §22).

        Only processes responses when registration parameters are configured.
        """
        if response.status_code == SIPStatus.OK and response.headers.get(
            "CSeq", ""
        ).split()[-1:] == [SIPMethod.REGISTER]:
            logger.info("Registration successful")
            self.registered()
            return
        if response.status_code in (
            SIPStatus.UNAUTHORIZED,
            SIPStatus.PROXY_AUTHENTICATION_REQUIRED,
        ):
            if not self.aor.user or not self.aor.password:
                logger.error(
                    "Auth challenge received but username/password are not configured"
                )
                return
            logger.debug(
                "Auth challenge received (%s), retrying with credentials",
                response.status_code,
            )
            is_proxy = response.status_code == SIPStatus.PROXY_AUTHENTICATION_REQUIRED
            challenge_key = "Proxy-Authenticate" if is_proxy else "WWW-Authenticate"
            params = self.parse_auth_challenge(response.headers.get(challenge_key, ""))
            realm = params.get("realm", "")
            nonce = params.get("nonce", "")
            opaque = params.get("opaque")
            algorithm = params.get("algorithm", DigestAlgorithm.SHA_256)
            qop_options = params.get("qop", "")
            qop = (
                DigestQoP.AUTH.value
                if DigestQoP.AUTH.value in qop_options.split(",")
                else None
            )
            nc = "00000001"
            cnonce = secrets.token_hex(8) if qop else None
            digest = self.digest_response(
                username=self.aor.user,
                password=self.aor.password,
                realm=realm,
                nonce=nonce,
                method=SIPMethod.REGISTER,
                uri=self.aor.host,
                algorithm=algorithm,
                qop=qop,
                nc=nc,
                cnonce=cnonce,
            )
            auth_value = (
                f'Digest username="{self.aor.user}", realm="{realm}", '
                f'nonce="{nonce}", uri="{self.aor.host}", '
                f'response="{digest}", algorithm="{algorithm}"'
            )
            if qop:
                auth_value += f', qop={qop}, nc={nc}, cnonce="{cnonce}"'
            if opaque:
                auth_value += f', opaque="{opaque}"'
            if is_proxy:
                asyncio.create_task(self.register(proxy_authorization=auth_value))
            else:
                asyncio.create_task(self.register(authorization=auth_value))
            return
        raise RegistrationError(f"{response.status_code} {response.phrase}")

    @property
    def contact(self) -> str:
        """Return a ``Contact:`` header value for this UA.

        The URI scheme mirrors `aor`: a ``sips:`` AOR produces a
        ``sips:`` Contact (the strongest TLS guarantee); a ``sip:`` AOR over
        TLS produces ``sip:`` with ``transport=tls``; plain TCP produces plain
        ``sip:``.

        When *ob* is ``True`` the ``ob`` URI parameter ([RFC 5626 §5]) is
        appended inside the angle brackets to advertise outbound keep-alive
        support to the registrar.

        [RFC 5626 §5]: https://datatracker.ietf.org/doc/html/rfc5626#section-5

        Args:
            user: SIP user part (e.g. ``"alice"``).  When provided the Contact
                is of the form ``<scheme:user@host:port>``; otherwise just
                ``<scheme:host:port>``.
            ob: Include the ``ob`` URI parameter (RFC 5626 §5) to indicate
                outbound keep-alive support.
        """
        host_port = f"{_format_host(self.local_address[0])}:{self.local_address[1]}"
        addr = f"{self.aor.user}@{host_port}" if self.aor.user else host_port
        ob_uri_param = ";ob"
        if self.aor.scheme == "sips":
            return f"<sips:{addr}{ob_uri_param}>"
        tls_param = ";transport=tls" if self.is_secure else ""
        return f"<sip:{addr}{tls_param}{ob_uri_param}>"

    def from_header(self, tag: str = "") -> str:
        """Return a ``From:`` header value for this UA."""
        from_uri = types.SipUri(
            user=self.aor.user, host=self.aor.host, scheme=self.aor.scheme
        )
        if tag:
            return f"{from_uri};tag={tag}"
        return str(from_uri)

    @property
    def to_header(self) -> str:
        """Return a ``To:`` header value for this UA."""
        params = {}
        if transport := self.aor.parameters.get("transport"):
            params["transport"] = transport
        to_uri = types.SipUri(
            user=self.aor.user,
            host=self.aor.host,
            scheme=self.aor.scheme,
            parameters=params,
        )
        return str(to_uri)

    async def register(
        self,
        authorization: str | None = None,
        proxy_authorization: str | None = None,
    ) -> None:
        """Send a REGISTER request to the registrar, optionally with credentials.

        The REGISTER Request-URI is the registrar URI derived from `aor`
        (RFC 3261 §10.2).  When an `outbound_proxy` is configured, the
        request is sent over the existing TLS/TCP connection to that proxy,
        which routes it to the registrar on our behalf.
        """
        self.cseq += 1
        if self.outbound_proxy:
            logger.debug(
                "Sending REGISTER via outbound proxy %s:%s to registrar %s (CSeq %s)",
                self.outbound_proxy[0],
                self.outbound_proxy[1],
                self.aor.host,
                self.cseq,
            )
        else:
            logger.debug(
                "Sending REGISTER to registrar %s (CSeq %s)",
                self.aor.host,
                self.cseq,
            )
        branch = f"{self.VIA_BRANCH_PREFIX}{secrets.token_hex(16)}"
        logger.debug("REGISTER Via branch: %s", branch)
        # Extract SIP user part from AOR (e.g. "sips:alice@example.com" -> "alice")
        headers = {
            "Via": f"SIP/2.0/{'TLS' if self.is_secure else 'TCP'} {_format_host(self.local_address[0])}:{self.local_address[1]};rport;branch={branch}",
            "From": self.from_header(),
            "To": self.to_header,
            "Call-ID": self.call_id,
            "CSeq": f"{self.cseq} {SIPMethod.REGISTER}",
            "Contact": self.contact,
            "Expires": "3600",  # 1 hour
            "Max-Forwards": "70",
            "Supported": "outbound",  # RFC 5626 §5 — outbound keep-alive support
        }
        if authorization is not None:
            headers["Authorization"] = authorization
        if proxy_authorization is not None:
            headers["Proxy-Authorization"] = proxy_authorization
        self.send(
            Request(
                method=SIPMethod.REGISTER,
                uri=types.SipUri(host=self.aor.host, scheme=self.aor.scheme),
                headers=headers,
            ),
        )

    def registered(self) -> None:
        """Handle a confirmed carrier registration. Override to react."""

    @staticmethod
    def parse_auth_challenge(header: str) -> dict[str, str]:
        """Parse Digest challenge parameters from a WWW-Authenticate/Proxy-Authenticate header."""
        _, _, params_str = header.partition(" ")
        params = {}
        for part in re.split(r",\s*(?=[a-zA-Z])", params_str):
            key, _, value = part.partition("=")
            if key.strip():
                params[key.strip()] = value.strip().strip('"')
        return params

    #: Map from `DigestAlgorithm` to the hashlib name.
    _DIGEST_HASH_NAME: typing.ClassVar[dict[str, str]] = {
        DigestAlgorithm.MD5: "md5",
        DigestAlgorithm.MD5_SESS: "md5",
        DigestAlgorithm.SHA_256: "sha256",
        DigestAlgorithm.SHA_256_SESS: "sha256",
        DigestAlgorithm.SHA_512_256: "sha512_256",
        DigestAlgorithm.SHA_512_256_SESS: "sha512_256",
    }

    @classmethod
    def digest_response(
        cls,
        *,
        username: str,
        password: str,
        realm: str,
        nonce: str,
        method: str,
        uri: str,
        algorithm: str = DigestAlgorithm.SHA_256,
        qop: str | None = None,
        nc: str = "00000001",
        cnonce: str | None = None,
    ) -> str:
        """Compute a SIP digest response per RFC 3261 §22 and RFC 8760.

        RFC 8760 deprecates MD5 and mandates support for SHA-256 and
        SHA-512-256.  The ``algorithm`` parameter selects the hash function;
        it defaults to ``SHA-256``.

        Raises:
            ValueError: If ``algorithm`` is not a recognised `DigestAlgorithm`,
                or if a ``*-sess`` algorithm is requested without a ``cnonce``.
        """
        try:
            hash_name = cls._DIGEST_HASH_NAME[algorithm]
        except KeyError:
            raise ValueError(f"Unsupported digest algorithm: {algorithm!r}") from None
        is_sess = algorithm.endswith("-sess")
        if is_sess and cnonce is None:
            raise ValueError(f"algorithm={algorithm!r} requires a cnonce value")

        def h(data: str) -> str:
            return hashlib.new(hash_name, data.encode()).hexdigest()

        ha1 = h(f"{username}:{realm}:{password}")
        if is_sess:
            ha1 = h(f"{ha1}:{nonce}:{cnonce}")
        ha2 = h(f"{method}:{uri}")
        if qop in (DigestQoP.AUTH, DigestQoP.AUTH_INT):
            return h(f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}")
        return h(f"{ha1}:{nonce}:{ha2}")

    def connection_lost(self, exc: Exception | None) -> None:
        """Handle a lost TLS/TCP connection."""
        if exc is not None:
            logger.exception("Connection lost", exc_info=exc)
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        self.transport = None
        self.disconnected_event.set()


#: Short alias for `SessionInitiationProtocol`.
SIP = SessionInitiationProtocol
