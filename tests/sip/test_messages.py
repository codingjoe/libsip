"""Tests for SIP message parsing and serialization."""

import pytest
from voip.sip.messages import Message, Request, Response


class TestSIPMessage:
    def test_parse__request(self):
        """Parse a SIP request from bytes."""
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds\r\n"
            b"\r\n"
        )
        result = Message.parse(data)
        assert isinstance(result, Request)
        assert result.method == "INVITE"
        assert result.uri == "sip:bob@biloxi.com"
        assert result.version == "SIP/2.0"
        assert result.headers == {
            "Via": "SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds"
        }
        assert result.body == b""

    def test_parse__request__with_body(self):
        """Parse a SIP request with a body from bytes."""
        data = b"INVITE sip:bob@biloxi.com SIP/2.0\r\nContent-Length: 4\r\n\r\ntest"
        result = Message.parse(data)
        assert isinstance(result, Request)
        assert result.body == b"test"

    def test_parse__response(self):
        """Parse a SIP response from bytes."""
        data = (
            b"SIP/2.0 200 OK\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds\r\n"
            b"\r\n"
        )
        result = Message.parse(data)
        assert isinstance(result, Response)
        assert result.status_code == 200
        assert result.reason == "OK"
        assert result.version == "SIP/2.0"
        assert result.headers == {
            "Via": "SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds"
        }
        assert result.body == b""

    def test_parse__response__with_body(self):
        """Parse a SIP response with a body from bytes."""
        data = b"SIP/2.0 200 OK\r\nContent-Length: 4\r\n\r\ntest"
        result = Message.parse(data)
        assert isinstance(result, Response)
        assert result.body == b"test"

    def test_parse__roundtrip_request(self):
        """Round-trip a SIP request through parse and bytes."""
        request = Request(
            method="REGISTER",
            uri="sip:registrar.biloxi.com",
            headers={"From": "sip:bob@biloxi.com"},
        )
        assert Message.parse(bytes(request)) == request

    def test_parse__roundtrip_response(self):
        """Round-trip a SIP response through parse and bytes."""
        response = Response(
            status_code=404,
            reason="Not Found",
            headers={"From": "sip:bob@biloxi.com"},
        )
        assert Message.parse(bytes(response)) == response

    def test_parse__skips_empty_header_lines(self):
        """Skip empty lines in the header section without raising."""
        # Extra \r\n before \r\n\r\n so the header_section has a trailing \r\n
        # which produces an empty string when split on \r\n.
        data = b"REGISTER sip:example.com SIP/2.0\r\nVia: SIP/2.0/UDP pc33\r\n\r\n\r\n"
        result = Message.parse(data)
        assert isinstance(result, Request)
        assert result.headers.get("Via") == "SIP/2.0/UDP pc33"

    def test_parse__skips_header_line_without_colon(self):
        """Skip header lines that contain no colon separator."""
        data = b"REGISTER sip:example.com SIP/2.0\r\nInvalidHeaderLine\r\n\r\n"
        result = Message.parse(data)
        assert isinstance(result, Request)
        assert "InvalidHeaderLine" not in result.headers

    def test_parse__raises_value_error_on_invalid_first_line(self):
        """Raise ValueError when the first line cannot be parsed as a request."""
        with pytest.raises(ValueError, match="Invalid SIP message"):
            Message.parse(b"TOOSHORT\r\n\r\n")


class TestRequest:
    def test_request__bytes(self):
        """Serialize a SIP request to bytes."""
        request = Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            headers={"Via": "SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds"},
        )
        assert bytes(request) == (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds\r\n"
            b"\r\n"
        )

    def test_request__bytes__with_body(self):
        """Serialize a SIP request with a body to bytes."""
        request = Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            headers={"Content-Length": "4"},
            body=b"test",
        )
        assert bytes(request) == (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\nContent-Length: 4\r\n\r\ntest"
        )


class TestResponse:
    def test_response__bytes(self):
        """Serialize a SIP response to bytes."""
        response = Response(
            status_code=200,
            reason="OK",
            headers={"Via": "SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds"},
        )
        assert bytes(response) == (
            b"SIP/2.0 200 OK\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds\r\n"
            b"\r\n"
        )

    def test_response__bytes__with_body(self):
        """Serialize a SIP response with a body to bytes (explicit Content-Length kept)."""
        response = Response(
            status_code=200,
            reason="OK",
            headers={"Content-Length": "4"},
            body=b"test",
        )
        assert bytes(response) == (b"SIP/2.0 200 OK\r\nContent-Length: 4\r\n\r\ntest")

    def test_response__bytes__with_body__auto_content_length(self):
        """Auto-calculate Content-Length when body is present and header is not set."""
        response = Response(status_code=200, reason="OK", body=b"test")
        serialized = bytes(response)
        assert b"Content-Length: 4" in serialized
        parsed = Message.parse(serialized)
        assert parsed.body == b"test"
