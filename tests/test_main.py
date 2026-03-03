"""Tests for the CLI commands."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

_click_testing = pytest.importorskip("click.testing")
CliRunner = _click_testing.CliRunner

# Stub out optional heavy dependencies so the CLI can be imported without them.
_WHISPER_STUBS = {
    "numpy": MagicMock(),
    "whisper": MagicMock(),
    "sip.whisper": MagicMock(WhisperCall=MagicMock),
}


def make_runner():
    return CliRunner()


class TestTranscribeCLI:
    def test_transcribe__missing_server_exits_with_error(self):
        """Exit with an error when SIP_SERVER is not provided."""
        from sip.__main__ import sip

        runner = make_runner()
        result = runner.invoke(
            sip, ["transcribe", "--aor=sip:u@h", "--username=u", "--password=p"]
        )
        assert result.exit_code != 0
        assert "server" in result.output.lower() or "SIP_SERVER" in result.output

    def test_transcribe__missing_aor_defaults_to_username_at_host(self):
        """Default AOR to sip:{username}@{server_host} when SIP_AOR is not provided."""
        from sip.__main__ import sip

        runner = make_runner()
        captured = {}

        async def fake_endpoint(factory, *, local_addr):
            protocol = factory()
            captured["aor"] = protocol.aor
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("sip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_datagram_endpoint = fake_endpoint
            runner.invoke(
                sip,
                [
                    "transcribe",
                    "--server=sip.example.com",
                    "--username=u",
                    "--password=p",
                ],
                catch_exceptions=False,
            )
        assert captured.get("aor") == "sip:u@sip.example.com"

    def test_transcribe__server_with_port_is_parsed(self):
        """Parse host and port from SIP_SERVER when a colon is present."""
        from sip.__main__ import sip

        runner = make_runner()
        captured = {}

        async def fake_endpoint(factory, *, local_addr):
            protocol = factory()
            captured["server_addr"] = protocol.server_address
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("sip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_datagram_endpoint = fake_endpoint
            runner.invoke(
                sip,
                [
                    "transcribe",
                    "--server=sip.carrier.example:5080",
                    "--aor=sip:user@carrier.example",
                    "--username=user",
                    "--password=pass",
                ],
                catch_exceptions=False,
            )
        assert captured.get("server_addr") == ("sip.carrier.example", 5080)

    def test_transcribe__server_without_port_defaults_to_5060(self):
        """Use port 5060 when SIP_SERVER has no port."""
        from sip.__main__ import sip

        runner = make_runner()
        captured = {}

        async def fake_endpoint(factory, *, local_addr):
            protocol = factory()
            captured["server_addr"] = protocol.server_address
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("sip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_datagram_endpoint = fake_endpoint
            runner.invoke(
                sip,
                [
                    "transcribe",
                    "--server=sip.carrier.example",
                    "--aor=sip:user@carrier.example",
                    "--username=user",
                    "--password=pass",
                ],
                catch_exceptions=False,
            )
        assert captured.get("server_addr") == ("sip.carrier.example", 5060)
