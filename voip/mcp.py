"""MCP server for VoIP actions.

This module exposes two MCP tools — [`say`][voip.mcp.say] and [`call`][voip.mcp.call] —
and a [`run`][voip.mcp.run] helper that handles all transport setup in one call,
mirroring the start-and-block pattern of [`mcp.run`][fastmcp.FastMCP.run].

Requires the ``mcp`` extra: ``pip install voip[mcp]``.
"""

import asyncio
import dataclasses
import threading
import typing

from fastmcp import Context, FastMCP
from mcp.types import SamplingMessage, TextContent

import voip
from voip import ai
from voip.ai import SayCall
from voip.sip import Dialog
from voip.sip.protocol import SessionInitiationProtocol
from voip.sip.types import SipURI, parse_uri
from voip.types import NetworkAddress

__all__ = [
    "mcp",
    "MCPAgentCall",
]

mcp = FastMCP(
    "VoIP",
    "Provide a set of tools to make phone calls.",
    version=voip.__version__,
    website_url="https://codingjoe.dev/VoIP/",
)

connection_pool = threading.local()


@dataclasses.dataclass(kw_only=True, slots=True)
class MCPAgentCall(ai.AgentCall):
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
                role=typing.cast(typing.Literal["user", "assistant"], msg["role"]),
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
    target_uri: SipURI = SipURI.parse(target)

    dialog = Dialog(sip=connection_pool.sip)
    await dialog.dial(target_uri, session_class=SayCall, text=prompt)


@mcp.tool
async def call(
    target: str,
    initial_prompt: str = "",
    system_prompt: str = None,
) -> str:
    """Call a phone number, hold a conversation, and return the transcript.

    Dials *target*, optionally speaks *initial_prompt*, then drives the
    conversation via [`MCPAgentCall`][voip.mcp.MCPAgentCall] (which samples
    the MCP client's language model for each reply).  Returns once the remote
    party hangs up.

    Args:
        target: Phone number or SIP URI to call, e.g. ``"tel:+1234567890"``
            or ``"sip:alice@example.com"``.
        initial_prompt: Opening message spoken when the call connects.
        system_prompt: System instruction passed to the language model.

    Returns:
        The full conversation transcript with ``Caller:`` / ``Agent:`` prefixes.
    """
    target_uri = parse_uri(target, connection_pool.sip.aor)
    dialog = Dialog(sip=connection_pool.sip)
    await dialog.dial(target_uri, session_class=MCPAgentCall)
    return dialog.session.transcript


async def run(
    fn: typing.Callable[[], None],
    aor: SipURI,
    *,
    no_verify_tls: bool = False,
    stun_server: NetworkAddress | None = None,
    transport=None,
) -> None:
    """Run the MCP agent."""
    connection_pool.sip: SessionInitiationProtocol = (
        await SessionInitiationProtocol.run(
            fn,
            aor,
            Dialog,
            no_verify_tls=no_verify_tls,
            stun_server=stun_server,
        )
    )
    await mcp.run_async(transport=transport)


if __name__ == "__main__":  # pragma: no cover
    mcp.run()
