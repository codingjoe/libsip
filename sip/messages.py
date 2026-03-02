"""SIP message types as defined by RFC 3261."""

from __future__ import annotations

import dataclasses


def _encode_message(first_line: str, headers: dict[str, str], body: bytes) -> bytes:
    header_lines = "".join(f"{name}: {value}\r\n" for name, value in headers.items())
    return f"{first_line}\r\n{header_lines}\r\n".encode() + body


@dataclasses.dataclass
class Request:
    """A SIP request message (RFC 3261 §7.1)."""

    method: str
    uri: str
    headers: dict[str, str]
    body: bytes = b""
    version: str = "SIP/2.0"

    def __bytes__(self) -> bytes:
        """Serialize to bytes."""
        return _encode_message(
            f"{self.method} {self.uri} {self.version}", self.headers, self.body
        )


@dataclasses.dataclass
class Response:
    """A SIP response message (RFC 3261 §7.2)."""

    status_code: int
    reason: str
    headers: dict[str, str]
    body: bytes = b""
    version: str = "SIP/2.0"

    def __bytes__(self) -> bytes:
        """Serialize to bytes."""
        return _encode_message(
            f"{self.version} {self.status_code} {self.reason}", self.headers, self.body
        )


def parse(data: bytes) -> Request | Response:
    """Parse a SIP message from raw bytes."""
    header_section, _, body = data.partition(b"\r\n\r\n")
    lines = header_section.decode().split("\r\n")
    first_line, *header_lines = lines
    headers = dict(line.split(": ", 1) for line in header_lines if ": " in line)
    parts = first_line.split(" ", 2)
    if first_line.startswith("SIP/"):
        version, status_code_str, reason = parts
        return Response(
            status_code=int(status_code_str),
            reason=reason,
            headers=headers,
            body=body,
            version=version,
        )
    method, uri, version = parts
    return Request(method=method, uri=uri, headers=headers, body=body, version=version)
