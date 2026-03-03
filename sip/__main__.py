#!/usr/bin/env python3
import asyncio
import logging
import time

from . import messages

try:
    import click
    from pygments import formatters, highlight

    from .lexers import SIPLexer
except ImportError as e:
    raise ImportError(
        "The SIP CLI requires needs to be installed via `pip install libsip[cli]`."
    ) from e


class ConsoleMessageProcessor:
    """Protocol mixin that prints messages to stdout."""

    def request_received(self, request: messages.Request, addr: tuple[str, int]):
        self.pprint(request, addr)
        super().request_received()

    def response_received(self, response: messages.Response, addr: tuple[str, int]):
        self.pprint(response, addr)
        super().request_received()

    @staticmethod
    def pprint(msg, addr):
        """Pretty print the message."""
        host = f"[{addr[0]}]" if ":" in addr[0] else addr[0]
        host = click.style(host, fg="green", bold=True)
        port = click.style(str(addr[1]), fg="yellow", bold=True)
        pretty_msg = highlight(str(msg), SIPLexer(), formatters.TerminalFormatter())
        click.echo(f"{host}:{port} - - [{time.asctime()}] {pretty_msg}")


@click.group()
@click.option("-v", "--verbose", count=True, help="Increase verbosity.")
def sip(verbose):
    """SIP command line interface."""
    logging.basicConfig(
        level=max(10, 10 * (2 - verbose)),
        format="%(levelname)s: [%(asctime)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


logger = logging.getLogger(__name__)

main = sip


@sip.command()
@click.option(
    "--model",
    default="base",
    envvar="WHISPER_MODEL",
    show_default=True,
    help="Whisper model size.",
)
@click.option(
    "--server",
    envvar="SIP_SERVER",
    required=True,
    metavar="HOST[:PORT]",
    help="SIP server address.",
)
@click.option(
    "--aor",
    envvar="SIP_AOR",
    required=True,
    metavar="SIP_AOR",
    help="SIP Address of Record.",
)
@click.option("--username", envvar="SIP_USERNAME", required=True, help="SIP username.")
@click.option("--password", envvar="SIP_PASSWORD", required=True, help="SIP password.")
@click.option(
    "--local-port", default=5060, show_default=True, help="Local UDP port to bind."
)
@click.option(
    "--stun-server",
    default="stun.l.google.com:19302",
    envvar="STUN_SERVER",
    show_default=True,
    help="STUN server for NAT traversal (HOST:PORT or 'none' to disable).",
)
def transcribe(model, server, aor, username, password, local_port, stun_server):
    """Register with a SIP carrier and transcribe incoming calls via Whisper."""
    from .calls import IncomingCall, RegisterProtocol  # noqa: PLC0415
    from .whisper import WhisperCall  # noqa: PLC0415

    if ":" in server:
        host, port_str = server.rsplit(":", 1)
        port = int(port_str)
    else:
        host, port = server, 5060
    server_addr = (host, port)

    if stun_server.lower() == "none":
        stun = None
    elif ":" in stun_server:
        stun_host, stun_port_str = stun_server.rsplit(":", 1)
        stun = (stun_host, int(stun_port_str))
    else:
        stun = (stun_server, 3478)

    class TranscribingCall(WhisperCall):
        def transcription_received(self, text: str) -> None:
            logger.info("Transcription: %s", text)
            click.echo(text)

    class TranscribingProtocol(RegisterProtocol):
        def registered(self) -> None:
            logger.info("Registered with %s — waiting for calls", server)
            click.echo(f"Registered with {server} — waiting for calls", err=True)

        def create_call(self, request, addr) -> TranscribingCall:
            return TranscribingCall(request, addr, self.send, model=model)

        def invite_received(self, call: IncomingCall, addr) -> None:
            click.echo(f"Incoming call from {call.caller}", err=True)
            asyncio.create_task(call.answer())

    async def run():
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(
            lambda: TranscribingProtocol(
                server_addr, aor, username, password, stun_server=stun
            ),
            local_addr=("0.0.0.0", local_port),  # noqa: S104
        )
        click.echo(f"Listening on port {local_port}…", err=True)
        await asyncio.Future()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":  # pragma: no cover
    sip()
