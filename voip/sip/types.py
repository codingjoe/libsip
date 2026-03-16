from __future__ import annotations

import dataclasses
import enum
import re

__all__ = ["CallerID", "DigestAlgorithm", "DigestQoP", "SipURI", "Status"]


@dataclasses.dataclass
class SipURI:
    """A parsed SIP or SIPS URI per [RFC 3261 §19.1].

    Format: ``sip:user:password@host:port;uri-parameters?headers``

    The `parse` classmethod decodes a raw SIP URI string into structured
    fields.  IPv6 addresses in the host part must be enclosed in square
    brackets per [RFC 2732] (e.g. ``sip:alice@[::1]:5060``); the stored
    `host` is the bare address without brackets.

    [RFC 3261 §19.1]: https://datatracker.ietf.org/doc/html/rfc3261#section-19.1
    [RFC 2732]: https://datatracker.ietf.org/doc/html/rfc2732

    Examples:
        >>> SipURI.parse("sip:alice@example.com")
        SipURI(scheme='sip', user='alice', host='example.com', ...)
        >>> SipURI.parse("sips:+15551234567@carrier.com:5061")
        SipURI(scheme='sips', user='+15551234567', host='carrier.com', port=5061, ...)
        >>> SipURI.parse("sip:alice@[::1]:5060")
        SipURI(scheme='sip', user='alice', host='::1', port=5060, ...)
    """

    scheme: str
    """URI scheme — ``"sip"`` or ``"sips"``."""
    user: str
    """SIP user part (phone number or username)."""
    host: str
    """Host as a bare string — no brackets for IPv6 addresses."""
    password: str | None = None
    """Optional password from the user-info component (``user:password@host``)."""
    port: int | None = None
    """Optional port number; ``None`` when not present in the URI."""
    uri_parameters: dict[str, str | None] = dataclasses.field(default_factory=dict)
    """URI parameters as a mapping of name → value (``None`` for flag parameters)."""
    headers: dict[str, str] = dataclasses.field(default_factory=dict)
    """SIP URI headers (``?Header=value``) as a mapping of name → value."""

    @classmethod
    def parse(cls, value: str) -> SipURI:
        """Parse a SIP or SIPS URI string into a `SipURI` instance.

        Implements the full ``sip:user:password@host:port;uri-parameters?headers``
        grammar from [RFC 3261 §19.1].  IPv6 host literals must be bracketed
        per [RFC 2732], e.g. ``sip:alice@[::1]:5060``.

        [RFC 3261 §19.1]: https://datatracker.ietf.org/doc/html/rfc3261#section-19.1
        [RFC 2732]: https://datatracker.ietf.org/doc/html/rfc2732

        Args:
            value: Raw SIP URI string.

        Returns:
            Parsed `SipURI` instance.

        Raises:
            ValueError: When the URI is malformed (missing scheme, missing
                ``user@host``, unclosed IPv6 bracket, empty host, or invalid port).
        """
        scheme, _, uri_remainder = value.partition(":")
        if not scheme or not uri_remainder:
            raise ValueError(
                f"Invalid SIP URI: {value!r}. Expected sip[s]:user@host[:port]."
            )

        # Strip SIP URI headers (?Header=value&...) — must come after parameters.
        uri_remainder, _, headers_str = uri_remainder.partition("?")
        headers = cls._parse_headers(headers_str)

        # Strip URI parameters (;param or ;param=value) — after hostport.
        uri_remainder, _, params_str = uri_remainder.partition(";")
        uri_parameters = cls._parse_uri_parameters(params_str)

        # Separate user-info (user:password) from hostport.
        if "@" in uri_remainder:
            user_info, _, hostport = uri_remainder.partition("@")
            user, _, raw_password = user_info.partition(":")
            password: str | None = raw_password or None
        else:
            raise ValueError(
                f"Invalid SIP URI: {value!r}. Missing user@host part."
            )

        # Parse hostport — IPv6 literals are enclosed in brackets.
        host, port = cls._parse_hostport(hostport, value)

        return cls(
            scheme=scheme,
            user=user,
            host=host,
            password=password,
            port=port,
            uri_parameters=uri_parameters,
            headers=headers,
        )

    @staticmethod
    def _parse_hostport(hostport: str, original: str) -> tuple[str, int | None]:
        """Parse ``host`` or ``[IPv6host]:port`` into ``(host, port)``.

        Args:
            hostport: The host-port portion of a SIP URI.
            original: The full original URI string (used in error messages only).

        Returns:
            Tuple of ``(bare_host, port)`` where ``port`` is ``None`` when absent.

        Raises:
            ValueError: When the bracket is unclosed, the host is empty, or the
                port is not a valid integer.
        """
        if hostport.startswith("["):
            bracket_end = hostport.find("]")
            if bracket_end == -1:
                raise ValueError(
                    f"Invalid SIP URI: {original!r}. Unclosed bracket in IPv6 address."
                )
            host = hostport[1:bracket_end]
            if not host:
                raise ValueError(
                    f"Invalid SIP URI: {original!r}. Empty host in IPv6 brackets."
                )
            remainder = hostport[bracket_end + 1 :]
            port_str = remainder.removeprefix(":")
            try:
                port: int | None = int(port_str) if port_str else None
            except ValueError:
                raise ValueError(
                    f"Invalid SIP URI: {original!r}. Invalid port: {port_str!r}."
                ) from None
        else:
            host, _, port_str = hostport.partition(":")
            if not host:
                raise ValueError(
                    f"Invalid SIP URI: {original!r}. Missing host."
                )
            try:
                port = int(port_str) if port_str else None
            except ValueError:
                raise ValueError(
                    f"Invalid SIP URI: {original!r}. Invalid port: {port_str!r}."
                ) from None
        return host, port

    @staticmethod
    def _parse_uri_parameters(params_str: str) -> dict[str, str | None]:
        """Parse semicolon-separated URI parameters into a dict.

        Args:
            params_str: Semicolon-separated parameter string (without the leading ``;``).

        Returns:
            Mapping of parameter name → value (``None`` for flag parameters).
        """
        params: dict[str, str | None] = {}
        if not params_str:
            return params
        for part in params_str.split(";"):
            if "=" in part:
                key, _, val = part.partition("=")
                params[key] = val
            elif part:
                params[part] = None
        return params

    @staticmethod
    def _parse_headers(headers_str: str) -> dict[str, str]:
        """Parse ampersand-separated SIP URI headers into a dict.

        Args:
            headers_str: Ampersand-separated header string (without the leading ``?``).

        Returns:
            Mapping of header name → value.
        """
        headers: dict[str, str] = {}
        if not headers_str:
            return headers
        for part in headers_str.split("&"):
            if "=" in part:
                key, _, val = part.partition("=")
                headers[key] = val
        return headers


class CallerID(str):
    """SIP From/To header value with structured access and privacy-safe repr.

    Behaves as a plain ``str`` so it is wire-format compatible and can be
    stored in header dicts unchanged.  ``repr()`` returns a short anonymized
    form that shows only the last four characters of the user part and the
    carrier domain — useful for log messages.

    Examples:
        >>> str(CallerID('"015114455910" <sip:015114455910@telefonica.de>;tag=abc'))
        '"015114455910" <sip:015114455910@telefonica.de>;tag=abc'
        >>> repr(CallerID('"015114455910" <sip:015114455910@telefonica.de>;tag=abc'))
        '****5910@telefonica.de'
        >>> repr(CallerID('sip:alice@example.com'))
        '*lice@example.com'
    """

    @property
    def display_name(self) -> str | None:
        """Display name from the From/To header, if present."""
        m = re.match(r'^"([^"]+)"\s*<|^([^<"]+?)\s*<', self)
        if m:
            return (m.group(1) or m.group(2) or "").strip() or None
        return None

    @property
    def user(self) -> str | None:
        """SIP user part (phone number or username)."""
        m = re.search(r"sips?:([^@>;\s]+)@", self)
        return m.group(1) if m else None

    @property
    def host(self) -> str | None:
        """Carrier domain extracted from the SIP URI."""
        m = re.search(r"sips?:[^@>;\s]+@([^>;)\s,]+)", self)
        return m.group(1) if m else None

    @property
    def tag(self) -> str | None:
        """Dialog tag parameter value, if present."""
        m = re.search(r";tag=([^\s;]+)", self)
        return m.group(1) if m else None

    def __repr__(self) -> str:
        """Anonymized label: last 4 chars of user + carrier domain."""
        user = self.display_name or self.user or ""
        host = self.host or ""
        masked = ("*" * max(0, len(user) - 4)) + user[-4:] if user else "****"
        return f"{masked}@{host}" if host else masked


Status = enum.IntEnum(
    value="Status",
    names={
        # 1xx Provisional
        "Trying": 100,
        "Ringing": 180,
        "Call Is Being Forwarded": 181,
        "Queued": 182,
        "Session Progress": 183,
        # 2xx Success
        "OK": 200,
        # 3xx Redirection
        "Multiple Choices": 300,
        "Moved Permanently": 301,
        "Moved Temporarily": 302,
        "Use Proxy": 305,
        "Alternative Service": 380,
        # 4xx Client Failure
        "Bad Request": 400,
        "Unauthorized": 401,
        "Payment Required": 402,
        "Forbidden": 403,
        "Not Found": 404,
        "Method Not Allowed": 405,
        "Not Acceptable": 406,
        "Proxy Authentication Required": 407,
        "Request Timeout": 408,
        "Gone": 410,
        "Request Entity Too Large": 413,
        "Request-URI Too Long": 414,
        "Unsupported Media Type": 415,
        "Unsupported URI Scheme": 416,
        "Bad Extension": 420,
        "Extension Required": 421,
        "Interval Too Brief": 423,
        "Temporarily Unavailable": 480,
        "Call/Transaction Does Not Exist": 481,
        "Loop Detected": 482,
        "Too Many Hops": 483,
        "Address Incomplete": 484,
        "Ambiguous": 485,
        "Busy Here": 486,
        "Request Terminated": 487,
        "Not Acceptable Here": 488,
        "Request Pending": 491,
        "Undecipherable": 493,
        # 5xx Server Failure
        "Server Internal Error": 500,
        "Not Implemented": 501,
        "Bad Gateway": 502,
        "Service Unavailable": 503,
        "Server Time-out": 504,
        "Version Not Supported": 505,
        "Message Too Large": 513,
        # 6xx Global Failure
        "Busy Everywhere": 600,
        "Decline": 603,
        "Does Not Exist Anywhere": 604,
    },
)


class DigestAlgorithm(enum.StrEnum):
    """Hash algorithms for SIP Digest Authentication (RFC 3261, RFC 8760).

    RFC 8760 deprecates MD5 in favour of SHA-256 and SHA-512-256.
    """

    MD5 = "MD5"
    MD5_SESS = "MD5-sess"
    SHA_256 = "SHA-256"
    SHA_256_SESS = "SHA-256-sess"
    SHA_512_256 = "SHA-512-256"
    SHA_512_256_SESS = "SHA-512-256-sess"


class DigestQoP(enum.StrEnum):
    """Quality of protection values for HTTP Digest Authentication (RFC 2617)."""

    AUTH = "auth"
    AUTH_INT = "auth-int"
