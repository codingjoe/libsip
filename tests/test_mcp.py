"""Tests for the MCP server (voip.mcp)."""

import asyncio
import dataclasses
import ipaddress
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

fastmcp = pytest.importorskip("fastmcp")
pytest.importorskip("faster_whisper")
pytest.importorskip("pocket_tts")

from fastmcp.exceptions import ToolError  # noqa: E402
from mcp.types import TextContent  # noqa: E402
from voip.mcp import (  # noqa: E402
    DEFAULT_STUN_SERVER,
    HangupDialog,
    MCPAgentCall,
    connect_rtp,
    connect_sip,
    read_aor,
    read_stun_server,
)
from voip.rtp import RealtimeTransportProtocol  # noqa: E402
from voip.sip.protocol import SessionInitiationProtocol  # noqa: E402
from voip.sip.types import SipURI  # noqa: E402
from voip.types import NetworkAddress  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_context(reply: str = "Hello!") -> MagicMock:
    """Return a mock FastMCP Context whose sample() returns *reply*."""
    ctx = MagicMock(spec=fastmcp.Context)
    result = MagicMock()
    result.text = reply
    ctx.sample = AsyncMock(return_value=result)
    return ctx


def make_mock_sip() -> MagicMock:
    """Return a mock SessionInitiationProtocol with a disconnected_event."""
    sip = MagicMock(spec=SessionInitiationProtocol)
    sip.disconnected_event = asyncio.Event()
    return sip


@dataclasses.dataclass
class FakeDatagramTransport:
    """Minimal UDP transport stub."""

    closed: bool = False

    def close(self) -> None:
        """Mark transport as closed."""
        self.closed = True


# ---------------------------------------------------------------------------
# Tests: read_aor
# ---------------------------------------------------------------------------


class TestReadAor:
    def test_read_aor__returns_parsed_uri(self, monkeypatch: pytest.MonkeyPatch):
        """Parses SIP_AOR from environment."""
        monkeypatch.setenv("SIP_AOR", "sip:alice@example.com:5060")
        aor = read_aor()
        assert isinstance(aor, SipURI)
        assert "alice" in str(aor)

    def test_read_aor__missing_raises_tool_error(self, monkeypatch: pytest.MonkeyPatch):
        """Raises ToolError when SIP_AOR is not set."""
        monkeypatch.delenv("SIP_AOR", raising=False)
        with pytest.raises(ToolError, match="SIP_AOR"):
            read_aor()


# ---------------------------------------------------------------------------
# Tests: read_stun_server
# ---------------------------------------------------------------------------


class TestReadStunServer:
    def test_read_stun_server__default(self, monkeypatch: pytest.MonkeyPatch):
        """Falls back to the Cloudflare STUN server when env var is absent."""
        monkeypatch.delenv("STUN_SERVER", raising=False)
        addr = read_stun_server()
        assert str(addr) == DEFAULT_STUN_SERVER

    def test_read_stun_server__custom(self, monkeypatch: pytest.MonkeyPatch):
        """Reads STUN server address from environment."""
        monkeypatch.setenv("STUN_SERVER", "stun.example.com:3479")
        addr = read_stun_server()
        assert isinstance(addr, NetworkAddress)


# ---------------------------------------------------------------------------
# Tests: connect_rtp
# ---------------------------------------------------------------------------


class TestConnectRTP:
    async def test_connect_rtp__ipv4(self):
        """Uses 0.0.0.0 as the bind address for IPv4 proxy addresses."""
        proxy_addr = NetworkAddress.parse("192.0.2.1:5060")
        fake_transport = FakeDatagramTransport()
        fake_rtp = MagicMock(spec=RealtimeTransportProtocol)

        loop = asyncio.get_running_loop()
        with patch.object(
            loop,
            "create_datagram_endpoint",
            new=AsyncMock(return_value=(fake_transport, fake_rtp)),
        ) as mock_create:
            t, proto = await connect_rtp(proxy_addr)
            assert t is fake_transport
            assert proto is fake_rtp
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["local_addr"] == ("0.0.0.0", 0)  # noqa: S104

    async def test_connect_rtp__ipv6(self):
        """Uses :: as the bind address for IPv6 proxy addresses."""
        proxy_addr = NetworkAddress(ipaddress.ip_address("::1"), 5060)
        fake_transport = FakeDatagramTransport()
        fake_rtp = MagicMock(spec=RealtimeTransportProtocol)

        loop = asyncio.get_running_loop()
        with patch.object(
            loop,
            "create_datagram_endpoint",
            new=AsyncMock(return_value=(fake_transport, fake_rtp)),
        ) as mock_create:
            await connect_rtp(proxy_addr)
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["local_addr"] == ("::", 0)

    async def test_connect_rtp__passes_stun_server(self):
        """Forwards the STUN server address to RealtimeTransportProtocol."""
        proxy_addr = NetworkAddress.parse("192.0.2.1:5060")
        stun = NetworkAddress.parse("stun.example.com:3478")
        fake_transport = FakeDatagramTransport()
        fake_rtp = MagicMock(spec=RealtimeTransportProtocol)

        loop = asyncio.get_running_loop()
        created_protocols = []

        async def fake_create(factory, **kwargs):
            protocol = factory()
            created_protocols.append(protocol)
            return fake_transport, fake_rtp

        with patch.object(loop, "create_datagram_endpoint", new=fake_create):
            await connect_rtp(proxy_addr, stun)

        assert len(created_protocols) == 1
        assert created_protocols[0].stun_server_address == stun


# ---------------------------------------------------------------------------
# Tests: connect_sip
# ---------------------------------------------------------------------------


class TestConnectSIP:
    async def test_connect_sip__with_tls(self):
        """Passes an SSL context when use_tls=True."""
        proxy_addr = NetworkAddress.parse("sip.example.com:5061")
        mock_protocol = make_mock_sip()
        # Pre-set the event so connect_sip returns immediately.
        mock_protocol.disconnected_event.set()

        loop = asyncio.get_running_loop()
        with patch.object(
            loop,
            "create_connection",
            new=AsyncMock(return_value=(MagicMock(), mock_protocol)),
        ) as mock_conn:
            await connect_sip(lambda: mock_protocol, proxy_addr, use_tls=True)
            _, kwargs = mock_conn.call_args
            assert kwargs["ssl"] is not None

    async def test_connect_sip__without_tls(self):
        """Passes ssl=None when use_tls=False."""
        proxy_addr = NetworkAddress.parse("sip.example.com:5060")
        mock_protocol = make_mock_sip()
        mock_protocol.disconnected_event.set()

        loop = asyncio.get_running_loop()
        with patch.object(
            loop,
            "create_connection",
            new=AsyncMock(return_value=(MagicMock(), mock_protocol)),
        ) as mock_conn:
            await connect_sip(lambda: mock_protocol, proxy_addr, use_tls=False)
            _, kwargs = mock_conn.call_args
            assert kwargs["ssl"] is None

    async def test_connect_sip__no_verify_tls(self):
        """Disables cert verification when no_verify_tls=True."""
        import ssl  # noqa: PLC0415

        proxy_addr = NetworkAddress.parse("sip.example.com:5061")
        mock_protocol = make_mock_sip()
        mock_protocol.disconnected_event.set()

        loop = asyncio.get_running_loop()
        captured_ssl = []

        async def fake_conn(factory, **kwargs):
            captured_ssl.append(kwargs.get("ssl"))
            return MagicMock(), mock_protocol

        with patch.object(loop, "create_connection", new=fake_conn):
            await connect_sip(
                lambda: mock_protocol,
                proxy_addr,
                use_tls=True,
                no_verify_tls=True,
            )

        assert captured_ssl[0] is not None
        assert captured_ssl[0].verify_mode == ssl.CERT_NONE

    async def test_connect_sip__awaits_disconnection(self):
        """Blocks until the protocol's disconnected_event fires."""
        proxy_addr = NetworkAddress.parse("sip.example.com:5060")
        mock_protocol = make_mock_sip()

        loop = asyncio.get_running_loop()
        with patch.object(
            loop,
            "create_connection",
            new=AsyncMock(return_value=(MagicMock(), mock_protocol)),
        ):
            # Signal disconnection after a short delay.
            async def disconnect_later() -> None:
                await asyncio.sleep(0)
                mock_protocol.disconnected_event.set()

            asyncio.create_task(disconnect_later())
            await connect_sip(lambda: mock_protocol, proxy_addr, use_tls=False)

        assert mock_protocol.disconnected_event.is_set()


# ---------------------------------------------------------------------------
# Tests: HangupDialog
# ---------------------------------------------------------------------------


class TestHangupDialog:
    def test_hangup_received__closes_sip(self):
        """Calls sip.close() when the remote party hangs up."""
        mock_sip = MagicMock(spec=SessionInitiationProtocol)
        d = HangupDialog(sip=mock_sip)
        d.hangup_received()
        mock_sip.close.assert_called_once()

    def test_hangup_received__no_sip(self):
        """Does not raise when sip is None."""
        d = HangupDialog()
        d.hangup_received()  # must not raise


# ---------------------------------------------------------------------------
# Helpers for MCPAgentCall
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FakeTransport:
    """Minimal asyncio.Transport stub."""

    closed: bool = False

    def write(self, data: bytes) -> None:
        """Discard outgoing bytes."""

    def close(self) -> None:
        """Mark as closed."""
        self.closed = True

    def get_extra_info(self, key: str, default=None):
        """Return socket metadata stubs."""
        match key:
            case "sockname":
                return ("127.0.0.1", 5004)
            case "peername":
                return ("192.0.2.1", 5004)
            case "ssl_object":
                return None
            case _:
                return default


def make_mcp_agent_call(ctx=None, **kwargs) -> MCPAgentCall:
    """Create an MCPAgentCall with mocked ML models and transport."""
    from unittest.mock import MagicMock  # noqa: PLC0415

    from voip.rtp import RealtimeTransportProtocol  # noqa: PLC0415
    from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415
    from voip.sip.dialog import Dialog  # noqa: PLC0415
    from voip.sip.types import CallerID  # noqa: PLC0415

    media = MediaDescription(
        media="audio",
        port=5004,
        proto="RTP/AVP",
        fmt=[RTPPayloadFormat(payload_type=0, encoding_name="pcmu", sample_rate=8000)],
    )
    mock_codec = MagicMock()
    mock_codec.payload_type = 0
    mock_codec.sample_rate_hz = 8000
    mock_codec.rtp_clock_rate_hz = 8000
    mock_codec.create_decoder.return_value = MagicMock()

    mock_tts = MagicMock()
    mock_tts.get_state_for_audio_prompt.return_value = {}
    mock_tts.sample_rate = 8000

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = ([], None)

    with patch("voip.codecs.get", return_value=mock_codec):
        return MCPAgentCall(
            ctx=ctx or make_mock_context(),
            rtp=MagicMock(spec=RealtimeTransportProtocol),
            dialog=Dialog(),
            media=media,
            caller=CallerID(""),
            tts_model=mock_tts,
            stt_model=mock_stt,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Tests: MCPAgentCall
# ---------------------------------------------------------------------------


class TestMCPAgentCall:
    def test_post_init__no_initial_prompt(self):
        """_messages starts empty when no initial_prompt is provided."""
        agent = make_mcp_agent_call()
        assert agent._messages == []

    def test_post_init__with_initial_prompt(self):
        """Adds the opening message to _messages and schedules TTS."""
        with patch("asyncio.create_task") as mock_task:
            agent = make_mcp_agent_call(initial_prompt="Hello there!")
        assert agent._messages == [
            {"role": "assistant", "content": "Hello there!"}
        ]
        mock_task.assert_called_once()

    def test_transcript__empty(self):
        """Returns an empty string when no messages exist."""
        agent = make_mcp_agent_call()
        assert agent.transcript == ""

    def test_transcript__formats_correctly(self):
        """Formats user messages with Caller: and agent messages with Agent:."""
        agent = make_mcp_agent_call()
        agent._messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        assert agent.transcript == "Caller: Hi\nAgent: Hello!"

    def test_transcription_received__appends_user_message(self):
        """Adds a user message to _messages and schedules respond()."""
        with patch("asyncio.create_task") as mock_task:
            agent = make_mcp_agent_call()
            agent.transcription_received("How are you?")

        assert agent._messages == [{"role": "user", "content": "How are you?"}]
        mock_task.assert_called_once()

    async def test_respond__calls_sample_and_speaks(self):
        """Calls ctx.sample and sends the reply as speech."""
        ctx = make_mock_context("I'm fine, thanks!")
        agent = make_mcp_agent_call(ctx=ctx)
        agent._messages = [{"role": "user", "content": "How are you?"}]

        agent.send_speech = AsyncMock()
        await agent.respond()

        ctx.sample.assert_called_once()
        sample_args = ctx.sample.call_args
        # Verify system_prompt is forwarded.
        assert "system_prompt" in sample_args.kwargs
        agent.send_speech.assert_called_once_with("I'm fine, thanks!")
        assert agent._messages[-1] == {
            "role": "assistant",
            "content": "I'm fine, thanks!",
        }

    async def test_respond__empty_reply_skipped(self):
        """Does not append an empty reply to _messages."""
        ctx = make_mock_context("   ")  # only whitespace
        agent = make_mcp_agent_call(ctx=ctx)
        agent._messages = [{"role": "user", "content": "Hi"}]
        agent.send_speech = AsyncMock()

        await agent.respond()

        agent.send_speech.assert_not_called()
        # Only the user message should be in _messages.
        assert len(agent._messages) == 1

    async def test_respond__none_text_skipped(self):
        """Does not raise when sample returns None text."""
        ctx = make_mock_context("")
        ctx.sample.return_value.text = None
        agent = make_mcp_agent_call(ctx=ctx)
        agent._messages = [{"role": "user", "content": "Hi"}]
        agent.send_speech = AsyncMock()

        await agent.respond()

        agent.send_speech.assert_not_called()

    async def test_respond__sampling_messages_include_history(self):
        """Forwards the full conversation history to ctx.sample."""
        from mcp.types import SamplingMessage  # noqa: PLC0415

        ctx = make_mock_context("Sure!")
        agent = make_mcp_agent_call(ctx=ctx)
        agent._messages = [
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What can you do?"},
        ]
        agent.send_speech = AsyncMock()

        await agent.respond()

        sampling_messages = ctx.sample.call_args.args[0]
        assert len(sampling_messages) == 2
        assert isinstance(sampling_messages[0], SamplingMessage)
        assert sampling_messages[0].role == "assistant"
        content = sampling_messages[0].content
        assert isinstance(content, TextContent)
        assert content.text == "Hi there!"


# ---------------------------------------------------------------------------
# Tests: say tool
# ---------------------------------------------------------------------------


class TestSayTool:
    async def test_say__missing_aor_raises(self, monkeypatch: pytest.MonkeyPatch):
        """Raises ToolError when SIP_AOR is not set."""
        monkeypatch.delenv("SIP_AOR", raising=False)
        from voip.mcp import say  # noqa: PLC0415

        ctx = make_mock_context()
        with pytest.raises(ToolError, match="SIP_AOR"):
            await say(ctx=ctx, target="tel:+1234567890", prompt="Hello")

    async def test_say__connects_and_dials(self, monkeypatch: pytest.MonkeyPatch):
        """Calls connect_rtp and connect_sip with correct parameters."""
        monkeypatch.setenv("SIP_AOR", "sip:alice@sip.example.com:5061;transport=TLS")
        monkeypatch.delenv("SIP_NO_VERIFY_TLS", raising=False)

        from voip.mcp import say  # noqa: PLC0415

        fake_rtp = MagicMock(spec=RealtimeTransportProtocol)
        fake_transport = FakeDatagramTransport()

        with (
            patch("voip.mcp.connect_rtp", new=AsyncMock(return_value=(fake_transport, fake_rtp))),
            patch("voip.mcp.connect_sip", new=AsyncMock()) as mock_sip,
        ):
            ctx = make_mock_context()
            await say(ctx=ctx, target="sip:bob@sip.example.com", prompt="Hi!")

        mock_sip.assert_called_once()
        call_kwargs = mock_sip.call_args
        assert call_kwargs.kwargs.get("use_tls") is True

    async def test_say__no_verify_tls(self, monkeypatch: pytest.MonkeyPatch):
        """Passes no_verify_tls=True when SIP_NO_VERIFY_TLS is set."""
        monkeypatch.setenv("SIP_AOR", "sip:alice@sip.example.com:5061;transport=TLS")
        monkeypatch.setenv("SIP_NO_VERIFY_TLS", "1")

        from voip.mcp import say  # noqa: PLC0415

        fake_rtp = MagicMock(spec=RealtimeTransportProtocol)
        fake_transport = FakeDatagramTransport()

        with (
            patch("voip.mcp.connect_rtp", new=AsyncMock(return_value=(fake_transport, fake_rtp))),
            patch("voip.mcp.connect_sip", new=AsyncMock()) as mock_sip,
        ):
            ctx = make_mock_context()
            await say(ctx=ctx, target="sip:bob@sip.example.com")

        call_kwargs = mock_sip.call_args
        assert call_kwargs.kwargs.get("no_verify_tls") is True


# ---------------------------------------------------------------------------
# Tests: call tool
# ---------------------------------------------------------------------------


class TestCallTool:
    async def test_call__missing_aor_raises(self, monkeypatch: pytest.MonkeyPatch):
        """Raises ToolError when SIP_AOR is not set."""
        monkeypatch.delenv("SIP_AOR", raising=False)
        from voip.mcp import call  # noqa: PLC0415

        ctx = make_mock_context()
        with pytest.raises(ToolError, match="SIP_AOR"):
            await call(ctx=ctx, target="tel:+1234567890")

    async def test_call__returns_empty_when_no_session(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Returns an empty string when no session was established."""
        monkeypatch.setenv("SIP_AOR", "sip:alice@sip.example.com:5061;transport=TLS")
        monkeypatch.delenv("SIP_NO_VERIFY_TLS", raising=False)

        from voip.mcp import call  # noqa: PLC0415

        fake_rtp = MagicMock(spec=RealtimeTransportProtocol)
        fake_transport = FakeDatagramTransport()

        with (
            patch("voip.mcp.connect_rtp", new=AsyncMock(return_value=(fake_transport, fake_rtp))),
            patch("voip.mcp.connect_sip", new=AsyncMock()),
        ):
            ctx = make_mock_context()
            result = await call(ctx=ctx, target="sip:bob@sip.example.com")

        assert result == ""

    async def test_call__connects_with_tls(self, monkeypatch: pytest.MonkeyPatch):
        """Passes use_tls=True when the AOR uses TLS transport."""
        monkeypatch.setenv("SIP_AOR", "sip:alice@sip.example.com:5061;transport=TLS")
        monkeypatch.delenv("SIP_NO_VERIFY_TLS", raising=False)

        from voip.mcp import call  # noqa: PLC0415

        fake_rtp = MagicMock(spec=RealtimeTransportProtocol)
        fake_transport = FakeDatagramTransport()

        with (
            patch("voip.mcp.connect_rtp", new=AsyncMock(return_value=(fake_transport, fake_rtp))),
            patch("voip.mcp.connect_sip", new=AsyncMock()) as mock_sip,
        ):
            ctx = make_mock_context()
            await call(ctx=ctx, target="sip:bob@sip.example.com")

        call_kwargs = mock_sip.call_args
        assert call_kwargs.kwargs.get("use_tls") is True

    async def test_call__no_verify_tls(self, monkeypatch: pytest.MonkeyPatch):
        """Passes no_verify_tls=True when SIP_NO_VERIFY_TLS is set."""
        monkeypatch.setenv("SIP_AOR", "sip:alice@sip.example.com:5061;transport=TLS")
        monkeypatch.setenv("SIP_NO_VERIFY_TLS", "true")

        from voip.mcp import call  # noqa: PLC0415

        fake_rtp = MagicMock(spec=RealtimeTransportProtocol)
        fake_transport = FakeDatagramTransport()

        with (
            patch("voip.mcp.connect_rtp", new=AsyncMock(return_value=(fake_transport, fake_rtp))),
            patch("voip.mcp.connect_sip", new=AsyncMock()) as mock_sip,
        ):
            ctx = make_mock_context()
            await call(ctx=ctx, target="sip:bob@sip.example.com")

        call_kwargs = mock_sip.call_args
        assert call_kwargs.kwargs.get("no_verify_tls") is True
