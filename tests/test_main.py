"""Tests for the CLI commands."""

from __future__ import annotations

import ipaddress
from unittest.mock import MagicMock

import pytest

pytest.importorskip("numpy")
_click_testing = pytest.importorskip("click.testing")
from voip.__main__ import voip  # noqa: E402

CliRunner = _click_testing.CliRunner

# Stub out optional heavy dependencies so the CLI can be imported without them.
_WHISPER_STUBS = {
    "numpy": MagicMock(),
    "whisper": MagicMock(),
    "av": MagicMock(),
    "faster_whisper": MagicMock(),
    "ollama": MagicMock(),
    "pocket_tts": MagicMock(),
    "voip.audio": MagicMock(
        EchoCall=MagicMock, VoiceActivityCall=MagicMock, AudioCall=MagicMock
    ),
    "voip.ai": MagicMock(TranscribeCall=MagicMock, AgentCall=MagicMock),
}


def make_runner():
    """Return a Click test runner."""
    return CliRunner()


def make_mock_transport(host: str = "127.0.0.1", port: int = 5060) -> MagicMock:
    """Return a MagicMock transport with a pre-configured sockname."""
    transport = MagicMock()
    transport.get_extra_info.return_value = (host, port)
    return transport


class TestParseStunServer:
    def test_parse_stun_server__none_disables_stun(self):
        """Return None when STUN server is not configured."""
        from voip.__main__ import _parse_stun_server

        assert _parse_stun_server(None, None, None) is None

    def test_parse_stun_server__string_none_disables_stun(self):
        """Return None when STUN server is explicitly disabled."""
        from voip.__main__ import _parse_stun_server

        assert _parse_stun_server(None, None, "none") is None

    def test_parse_stun_server__without_port_uses_stun_default(self):
        """Return port 3478 when no port is specified for STUN."""
        from voip.__main__ import _parse_stun_server

        assert _parse_stun_server(None, None, "stun.example.com") == (
            "stun.example.com",
            3478,
        )


class TestParseHostport:
    def test_parse_hostport__bracketed_ipv6_without_port_uses_default(self):
        """Return default port and IPv6Address when bracketed IPv6 address has no port."""
        from voip.__main__ import _parse_hostport

        assert _parse_hostport(None, None, "[::1]", default_port=5061) == (
            ipaddress.IPv6Address("::1"),
            5061,
        )

    def test_parse_hostport__bracketed_ipv6_with_port(self):
        """Return explicit port and IPv6Address when bracketed IPv6 address includes a port."""
        from voip.__main__ import _parse_hostport

        assert _parse_hostport(None, None, "[::1]:5061") == (
            ipaddress.IPv6Address("::1"),
            5061,
        )

    def test_parse_hostport__unbracketed_ipv6_raises_bad_parameter(self):
        """Raise BadParameter when an unbracketed IPv6 literal is given."""
        import click
        from voip.__main__ import _parse_hostport

        with pytest.raises(click.BadParameter, match="enclosed in brackets"):
            _parse_hostport(None, None, "::1")


class TestVoIPCommand:
    def test_voip__verbose_flag(self):
        """Accept -v flag without error."""
        result = make_runner().invoke(voip, ["-v", "--help"])
        assert result.exit_code == 0
