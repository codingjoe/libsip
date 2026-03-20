from __future__ import annotations

import ipaddress

import pytest
from voip.sip import SipUri
from voip.sip.messages import Response
from voip.sip.types import (
    InviteServerTransaction,
    NonInviteServerTransaction,
    TransactionState,
)

class TestSipUri:
    @pytest.mark.parametrize(
        "uri_str, expected_uri_obj",
        [
            # domain
            (
                "sip:alice@example.com",
                SipUri(scheme="sip", user="alice", host="example.com", port=5060),
            ),
            (
                "sips:alice@example.com",
                SipUri(scheme="sips", user="alice", host="example.com", port=5061),
            ),
            (
                "sip:alice@example.com:4050",
                SipUri(scheme="sip", user="alice", host="example.com", port=4050),
            ),
            (
                "sips:alice@example.com:4051",
                SipUri(scheme="sips", user="alice", host="example.com", port=4051),
            ),
            # ipv4
            (
                "sip:alice@192.168.1.1",
                SipUri(scheme="sip", user="alice", host="192.168.1.1", port=5060),
            ),
            (
                "sips:alice@192.168.1.1",
                SipUri(scheme="sips", user="alice", host="192.168.1.1", port=5061),
            ),
            (
                "sip:alice@192.168.1.1:4050",
                SipUri(scheme="sip", user="alice", host="192.168.1.1", port=4050),
            ),
            (
                "sips:alice@192.168.1.1:4051",
                SipUri(scheme="sips", user="alice", host="192.168.1.1", port=4051),
            ),
            # ipv6
            (
                "sip:alice@[::1]",
                SipUri(scheme="sip", user="alice", host="::1", port=5060),
            ),
            (
                "sips:alice@[::1]",
                SipUri(scheme="sips", user="alice", host="::1", port=5061),
            ),
            (
                "sip:alice@[::1]:4050",
                SipUri(scheme="sip", user="alice", host="::1", port=4050),
            ),
            (
                "sips:alice@[::1]:4051",
                SipUri(scheme="sips", user="alice", host="::1", port=4051),
            ),
            # uri-parameters
            (
                "sip:alice@example.com;transport=tcp",
                SipUri(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    parameters={"transport": "tcp"},
                ),
            ),
            (
                "sip:alice@example.com;transport=udp;ttl=15",
                SipUri(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    parameters={"transport": "udp", "ttl": "15"},
                ),
            ),
            # headers
            (
                "sip:alice@example.com?foo=bar",
                SipUri(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    headers={"foo": "bar"},
                ),
            ),
            (
                "sip:alice@example.com?tag=12345&foo=bar",
                SipUri(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    headers={"tag": "12345", "foo": "bar"},
                ),
            ),
            (
                r"sip:%61lice@atlanta.com;transport=TCP",
                SipUri(
                    scheme="sip",
                    user="alice",
                    host="atlanta.com",
                    port=5060,
                    parameters={"transport": "TCP"},
                ),
            ),
            (
                r"sip:atlanta.com;method=REGISTER?to=alice%40atlanta.com",
                SipUri(
                    scheme="sip",
                    user=None,
                    host="atlanta.com",
                    port=5060,
                    parameters={"method": "REGISTER"},
                    headers={"to": "alice@atlanta.com"},
                ),
            ),
        ],
    )
    def test_parse_valid(self, uri_str, expected_uri_obj):
        """Parse scheme, user, host and optional port from a valid SIP URI."""
        assert SipUri.parse(uri_str) == expected_uri_obj

    @pytest.mark.parametrize(
        "uri_str",
        [
            "http://example.com",  # wrong scheme
            "sip:@example.com",  # missing user
            "sip:@example.com:invalid-port",  # non-integer port
        ],
    )
    def test_parse_invalid(self, uri_str):
        """Raise ValueError when parsing an invalid SIP URI."""
        with pytest.raises(ValueError):
            SipUri.parse(uri_str)

    @pytest.mark.parametrize(
        "uri_obj, expected_uri_str",
        [
            (
                SipUri(scheme="sip", user="alice", host="example.com", port=5061),
                "sip:alice@example.com:5061",
            ),
            (
                SipUri(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    parameters={"transport": "TCP"},
                ),
                "sip:alice@example.com:5060;transport=TCP",
            ),
            (
                SipUri(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    headers={"foo": "bar"},
                ),
                "sip:alice@example.com:5060?foo=bar",
            ),
            (
                SipUri(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    parameters={"transport": "TCP"},
                    headers={"foo": "bar"},
                ),
                "sip:alice@example.com:5060;transport=TCP?foo=bar",
            ),
            # IPv6
            (
                SipUri(scheme="sip", user="alice", host="::1", port=5060),
                "sip:alice@[::1]:5060",
            ),
            (
                SipUri(
                    scheme="sip",
                    user="alice",
                    host=ipaddress.IPv6Address("::1"),
                    port=5060,
                ),
                "sip:alice@[::1]:5060",
            ),
            # IPv4
            (
                SipUri(scheme="sip", user="alice", host="127.0.0.1", port=5060),
                "sip:alice@127.0.0.1:5060",
            ),
            (
                SipUri(
                    scheme="sip",
                    user="alice",
                    host=ipaddress.IPv4Address("127.0.0.1"),
                    port=5060,
                ),
                "sip:alice@127.0.0.1:5060",
            ),
            # password in user-info
            (
                SipUri(
                    scheme="sip",
                    user="alice",
                    password="secret",  # noqa: S106
                    host="example.com",
                    port=5060,
                ),
                "sip:alice:secret@example.com:5060",
            ),
            # flag URI parameter (value=None) in __str__
            (
                SipUri(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    parameters={"lr": None},
                ),
                "sip:alice@example.com:5060;lr",
            ),
        ],
    )
    def test_str(self, uri_obj, expected_uri_str):
        """Test string representation of SipUri objects."""
        assert str(uri_obj) == expected_uri_str

    @pytest.mark.parametrize(
        "uri_str, expected_uri_obj",
        [
            # flag URI parameter (;lr with no value)
            (
                "sip:alice@example.com;lr",
                SipUri(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    parameters={"lr": None},
                ),
            ),
            # header without '=' value
            (
                "sip:alice@example.com?Subject",
                SipUri(
                    scheme="sip",
                    user="alice",
                    host="example.com",
                    port=5060,
                    headers={"Subject": ""},
                ),
            ),
        ],
    )
    def test_parse__flag_parameter_and_valueless_header(
        self, uri_str, expected_uri_obj
    ):
        """Parse flag URI parameters and valueless headers."""
        assert SipUri.parse(uri_str) == expected_uri_obj


def _ok() -> Response:
    return Response(status_code=200, phrase="OK")


def _trying() -> Response:
    return Response(status_code=100, phrase="Trying")


class TestInviteServerTransaction:
    def test_initial_state(self):
        """INVITE server transaction starts in PROCEEDING."""
        txn = InviteServerTransaction(branch="z9hG4bK1", call_id="c1", method="INVITE")
        assert txn.state == TransactionState.PROCEEDING
        assert txn.last_response is None

    def test_proceeding__records_provisional(self):
        """proceeding() stores 1xx response without changing state."""
        txn = InviteServerTransaction(branch="z9hG4bK1", call_id="c1", method="INVITE")
        r = _trying()
        txn.proceeding(r)
        assert txn.last_response is r
        assert txn.state == TransactionState.PROCEEDING

    def test_proceeding__invalid_state(self):
        """proceeding() raises when not in PROCEEDING."""
        txn = InviteServerTransaction(branch="z9hG4bK1", call_id="c1", method="INVITE")
        txn.complete(_ok())
        with pytest.raises(ValueError):
            txn.proceeding(_trying())

    def test_complete__transitions_to_completed(self):
        """complete() records final response and moves to COMPLETED."""
        txn = InviteServerTransaction(branch="z9hG4bK1", call_id="c1", method="INVITE")
        r = _ok()
        txn.complete(r)
        assert txn.last_response is r
        assert txn.state == TransactionState.COMPLETED

    def test_complete__invalid_state(self):
        """complete() raises when not in PROCEEDING."""
        txn = InviteServerTransaction(branch="z9hG4bK1", call_id="c1", method="INVITE")
        txn.complete(_ok())
        with pytest.raises(ValueError):
            txn.complete(_ok())

    def test_confirm__transitions_to_confirmed(self):
        """confirm() moves from COMPLETED to CONFIRMED."""
        txn = InviteServerTransaction(branch="z9hG4bK1", call_id="c1", method="INVITE")
        txn.complete(_ok())
        txn.confirm()
        assert txn.state == TransactionState.CONFIRMED

    def test_confirm__invalid_state(self):
        """confirm() raises when not in COMPLETED."""
        txn = InviteServerTransaction(branch="z9hG4bK1", call_id="c1", method="INVITE")
        with pytest.raises(ValueError):
            txn.confirm()

    def test_terminate(self):
        """terminate() moves to TERMINATED from any state."""
        txn = InviteServerTransaction(branch="z9hG4bK1", call_id="c1", method="INVITE")
        txn.terminate()
        assert txn.state == TransactionState.TERMINATED


class TestNonInviteServerTransaction:
    def test_initial_state(self):
        """Non-INVITE server transaction starts in TRYING."""
        txn = NonInviteServerTransaction(branch="z9hG4bK2", call_id="c2", method="BYE")
        assert txn.state == TransactionState.TRYING
        assert txn.last_response is None

    def test_proceeding__from_trying(self):
        """proceeding() transitions TRYING → PROCEEDING."""
        txn = NonInviteServerTransaction(branch="z9hG4bK2", call_id="c2", method="BYE")
        r = _trying()
        txn.proceeding(r)
        assert txn.last_response is r
        assert txn.state == TransactionState.PROCEEDING

    def test_proceeding__from_proceeding(self):
        """proceeding() is idempotent in PROCEEDING state."""
        txn = NonInviteServerTransaction(branch="z9hG4bK2", call_id="c2", method="BYE")
        txn.proceeding(_trying())
        r2 = _trying()
        txn.proceeding(r2)
        assert txn.last_response is r2
        assert txn.state == TransactionState.PROCEEDING

    def test_proceeding__invalid_state(self):
        """proceeding() raises when in COMPLETED."""
        txn = NonInviteServerTransaction(branch="z9hG4bK2", call_id="c2", method="BYE")
        txn.complete(_ok())
        with pytest.raises(ValueError):
            txn.proceeding(_trying())

    def test_complete__from_trying(self):
        """complete() transitions TRYING → COMPLETED."""
        txn = NonInviteServerTransaction(branch="z9hG4bK2", call_id="c2", method="BYE")
        r = _ok()
        txn.complete(r)
        assert txn.last_response is r
        assert txn.state == TransactionState.COMPLETED

    def test_complete__from_proceeding(self):
        """complete() transitions PROCEEDING → COMPLETED."""
        txn = NonInviteServerTransaction(branch="z9hG4bK2", call_id="c2", method="BYE")
        txn.proceeding(_trying())
        r = _ok()
        txn.complete(r)
        assert txn.last_response is r
        assert txn.state == TransactionState.COMPLETED

    def test_complete__invalid_state(self):
        """complete() raises when in COMPLETED."""
        txn = NonInviteServerTransaction(branch="z9hG4bK2", call_id="c2", method="BYE")
        txn.complete(_ok())
        with pytest.raises(ValueError):
            txn.complete(_ok())

    def test_terminate(self):
        """terminate() moves to TERMINATED from any state."""
        txn = NonInviteServerTransaction(branch="z9hG4bK2", call_id="c2", method="BYE")
        txn.terminate()
        assert txn.state == TransactionState.TERMINATED
