"""MCP server for VoIP actions.

This module exposes two MCP tools — [`say`][voip.mcp.say] and [`call`][voip.mcp.call] —
and two public transport factory functions — [`connect_rtp`][voip.mcp.connect_rtp] and
[`connect_sip`][voip.mcp.connect_sip] — that can be used independently to build custom
VoIP integrations.

The factory functions follow the same *start-and-block* pattern as
[`mcp.run`][fastmcp.FastMCP.run]: call them with the desired parameters and they handle
connection setup, the call lifecycle, and teardown internally.

Requires the ``mcp`` extra: ``pip install voip[mcp]``.
"""

import asyncio
import collections.abc
import dataclasses
import ipaddress
import os
import ssl
import typing

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import SamplingMessage, TextContent

import voip
from voip.ai import SayCall, TranscribeCall, TTSMixin
from voip.rtp import RealtimeTransportProtocol
from voip.sip import dialog
from voip.sip.protocol import SessionInitiationProtocol
from voip.sip.types import SipURI, parse_uri
from voip.types import NetworkAddress

__all__ = [
    "mcp",
    "connect_rtp",
    "connect_sip",
    "HangupDialog",
    "MCPAgentCall",
    "read_aor",
    "read_stun_server",
    "DEFAULT_STUN_SERVER",
    "DEFAULT_SYSTEM_PROMPT",
]

#: Default STUN server used when ``STUN_SERVER`` env var is absent.
DEFAULT_STUN_SERVER: typing.Final[str] = "stun.cloudflare.com:3478"

#: Default system prompt for [`MCPAgentCall`][voip.mcp.MCPAgentCall].
DEFAULT_SYSTEM_PROMPT: typing.Final[str] = (
    "You are a person on a phone call."
    " Keep your answers very brief and conversational."
    " YOU MUST NEVER USE NON-VERBAL CHARACTERS IN YOUR RESPONSES!"
)

mcp = FastMCP(
    "VoIP",
    "Provide a set of tools to make phone calls.",
    version=voip.__version__,
    website_url="https://codingjoe.dev/VoIP/",
)


async def connect_rtp(
    proxy_addr: NetworkAddress,
    stun_server: NetworkAddress | None = None,
) -> tuple[asyncio.DatagramTransport, RealtimeTransportProtocol]:
    """Create and connect an RTP transport.

    This factory function mirrors the simplicity of
    [`mcp.run`][fastmcp.FastMCP.run]: provide the target address and it
    handles binding, STUN negotiation, and returns a ready-to-use transport.

    Args:
        proxy_addr: Address of the SIP proxy, used to pick the correct IP family.
        stun_server: Optional STUN server address for NAT traversal.

    Returns:
        A ``(transport, protocol)`` tuple for the new RTP endpoint.
    """
    loop = asyncio.get_running_loop()
    rtp_bind_address = "::" if isinstance(proxy_addr[0], ipaddress.IPv6Address) else "0.0.0.0"  # noqa: S104
    return await loop.create_datagram_endpoint(
        lambda: RealtimeTransportProtocol(stun_server_address=stun_server),
        local_addr=(rtp_bind_address, 0),
    )


async def connect_sip(
    factory: collections.abc.Callable[[], SessionInitiationProtocol],
    proxy_addr: NetworkAddress,
    *,
    use_tls: bool = True,
    no_verify_tls: bool = False,
) -> None:
    """Connect a SIP transport and block until the session ends.

    Like [`mcp.run`][fastmcp.FastMCP.run], this is a *start-and-block* call:
    it establishes the TCP/TLS connection to the SIP proxy using `factory`,
    registers the user agent, and suspends until the connection is closed
    (typically after a call ends and the transport is shut down).

    Args:
        factory: Callable that returns a new
            [`SessionInitiationProtocol`][voip.sip.protocol.SessionInitiationProtocol]
            instance for each connection attempt.
        proxy_addr: Address of the SIP proxy to connect to.
        use_tls: Whether to wrap the connection in TLS. Defaults to ``True``.
        no_verify_tls: Disable TLS certificate verification. Insecure; for
            testing only. Defaults to ``False``.
    """
    loop = asyncio.get_running_loop()
    ssl_context: ssl.SSLContext | None = None
    if use_tls:
        ssl_context = ssl.create_default_context()
        if no_verify_tls:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
    _, protocol = await loop.create_connection(
        factory,
        host=str(proxy_addr[0]),
        port=proxy_addr[1],
        ssl=ssl_context,
    )
    await protocol.disconnected_event.wait()


class HangupDialog(dialog.Dialog):
    """Dialog that closes the SIP transport when the remote party hangs up.

    Use this in outbound call factories to ensure the SIP connection is
    released after a remote BYE, which in turn unblocks
    [`connect_sip`][voip.mcp.connect_sip].
    """

    def hangup_received(self) -> None:
        """Close the SIP transport on receiving a remote BYE."""
        if self.sip is not None:
            self.sip.close()


@dataclasses.dataclass(kw_only=True, slots=True)
class MCPAgentCall(TTSMixin, TranscribeCall):
    """Agent call that generates voice responses via MCP sampling.

    Transcribes the remote party's speech with
    [Whisper][voip.ai.TranscribeCall], then forwards the conversation
    history to the MCP client's language model via
    [`Context.sample`][fastmcp.Context.sample] and speaks the reply
    using [Pocket TTS][voip.ai.TTSMixin].

    Args:
        ctx: The FastMCP [`Context`][fastmcp.Context] used for LLM sampling.
        system_prompt: System instruction forwarded to the language model.
        initial_prompt: Opening message spoken as soon as the call connects.
    """

    ctx: Context
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    initial_prompt: str = ""

    _messages: list[dict[str, str]] = dataclasses.field(
        init=False, repr=False, default_factory=list
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.initial_prompt:
            self._messages.append(
                {"role": "assistant", "content": self.initial_prompt}
            )
            asyncio.create_task(self.send_speech(self.initial_prompt))

    @property
    def transcript(self) -> str:
        """Formatted conversation transcript.

        Returns:
            Each turn on its own line, prefixed with ``Caller:`` or ``Agent:``.
        """
        lines = []
        for msg in self._messages:
            role = "Caller" if msg["role"] == "user" else "Agent"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def transcription_received(self, text: str) -> None:
        """Handle a transcription chunk and schedule an LLM response.

        Args:
            text: Transcribed speech from the remote party (already stripped).
        """
        self._messages.append({"role": "user", "content": text})
        asyncio.create_task(self.respond())

    async def respond(self) -> None:
        """Sample the MCP client LLM and speak the reply."""
        sampling_messages = [
            SamplingMessage(
                role=typing.cast(
                    typing.Literal["user", "assistant"], msg["role"]
                ),
                content=TextContent(type="text", text=msg["content"]),
            )
            for msg in self._messages
        ]
        result = await self.ctx.sample(
            sampling_messages,
            system_prompt=self.system_prompt,
        )
        if result.text and (reply := result.text.strip()):
            self._messages.append({"role": "assistant", "content": reply})
            await self.send_speech(reply)


def read_aor() -> SipURI:
    """Read and parse the SIP AOR from the ``SIP_AOR`` environment variable.

    Returns:
        Parsed [`SipURI`][voip.sip.types.SipURI].

    Raises:
        ToolError: When ``SIP_AOR`` is not set.
    """
    aor_str = os.environ.get("SIP_AOR")
    if not aor_str:
        raise ToolError("SIP_AOR environment variable is not set.")
    return SipURI.parse(aor_str)


def read_stun_server() -> NetworkAddress:
    """Read the STUN server address from the environment.

    Falls back to [`DEFAULT_STUN_SERVER`][voip.mcp.DEFAULT_STUN_SERVER] when
    ``STUN_SERVER`` is unset.

    Returns:
        Parsed [`NetworkAddress`][voip.types.NetworkAddress].
    """
    return NetworkAddress.parse(
        os.environ.get("STUN_SERVER", DEFAULT_STUN_SERVER)
    )


@mcp.tool
async def say(ctx: Context, target: str, prompt: str = "") -> None:
    """Call a phone number and speak a message.

    Dials *target*, synthesises *prompt* as speech via Pocket TTS, then hangs
    up automatically once the message has been delivered.

    Args:
        ctx: FastMCP context (injected automatically by the framework).
        target: Phone number or SIP URI to call, e.g. ``"tel:+1234567890"``
            or ``"sip:alice@example.com"``.
        prompt: Text to speak during the call.
    """
    aor = read_aor()
    stun_server = read_stun_server()
    no_verify_tls = os.environ.get("SIP_NO_VERIFY_TLS", "").lower() in ("1", "true")
    target_uri = parse_uri(target, aor)

    _, rtp_protocol = await connect_rtp(aor.maddr, stun_server)

    @dataclasses.dataclass(kw_only=True, slots=True)
    class OutboundProtocol(SessionInitiationProtocol):
        dial_target: SipURI

        def on_registered(self) -> None:
            d = HangupDialog(sip=self)
            asyncio.create_task(
                d.dial(self.dial_target, session_class=SayCall, text=prompt)
            )

    await connect_sip(
        lambda: OutboundProtocol(aor=aor, rtp=rtp_protocol, dial_target=target_uri),
        aor.maddr,
        use_tls=aor.transport == "TLS",
        no_verify_tls=no_verify_tls,
    )


@mcp.tool
async def call(
    ctx: Context,
    target: str,
    initial_prompt: str = "",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Call a phone number, hold a conversation, and return the transcript.

    Dials *target*, optionally speaks *initial_prompt*, then drives the
    conversation via [`MCPAgentCall`][voip.mcp.MCPAgentCall] (which samples
    the MCP client's language model for each reply).  Returns once the remote
    party hangs up.

    Args:
        ctx: FastMCP context (injected automatically by the framework).
        target: Phone number or SIP URI to call, e.g. ``"tel:+1234567890"``
            or ``"sip:alice@example.com"``.
        initial_prompt: Opening message spoken when the call connects.
        system_prompt: System instruction passed to the language model.

    Returns:
        The full conversation transcript with ``Caller:`` / ``Agent:`` prefixes.
    """
    aor = read_aor()
    stun_server = read_stun_server()
    no_verify_tls = os.environ.get("SIP_NO_VERIFY_TLS", "").lower() in ("1", "true")
    target_uri = parse_uri(target, aor)

    _, rtp_protocol = await connect_rtp(aor.maddr, stun_server)
    sessions: list[MCPAgentCall] = []

    @dataclasses.dataclass(kw_only=True, slots=True)
    class CallSession(MCPAgentCall):
        def __post_init__(self) -> None:
            super().__post_init__()
            sessions.append(self)

    @dataclasses.dataclass(kw_only=True, slots=True)
    class OutboundProtocol(SessionInitiationProtocol):
        dial_target: SipURI

        def on_registered(self) -> None:
            d = HangupDialog(sip=self)
            asyncio.create_task(
                d.dial(
                    self.dial_target,
                    session_class=CallSession,
                    ctx=ctx,
                    system_prompt=system_prompt,
                    initial_prompt=initial_prompt,
                )
            )

    await connect_sip(
        lambda: OutboundProtocol(aor=aor, rtp=rtp_protocol, dial_target=target_uri),
        aor.maddr,
        use_tls=aor.transport == "TLS",
        no_verify_tls=no_verify_tls,
    )
    return sessions[0].transcript if sessions else ""


if __name__ == "__main__":  # pragma: no cover
    mcp.run()
