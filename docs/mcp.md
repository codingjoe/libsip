# MCP Server

The VoIP library ships with an [MCP](https://modelcontextprotocol.io/) server that
exposes phone-calling capabilities as tools an LLM agent can invoke.

## Tools

### `outbound_message`

Dial a number, synthesise a text message using Pocket TTS, and hang up.

```python
await server.outbound_message(
    target="sip:+15551234567@carrier.example.com",
    text="Your parcel has been delivered.",
)
```

### `make_call`

Dial a number and conduct an interactive voice conversation.
LLM inference is delegated to the MCP client via
[sampling](https://modelcontextprotocol.io/docs/concepts/sampling), so no local model is required.
When the remote party hangs up, the full transcript is summarised and returned.

```python
summary = await server.make_call(
    target="sip:+15551234567@carrier.example.com",
    initial_prompt="Hello, this is the AI assistant from Acme Corp.",
    system_prompt=(
        "You are a friendly customer-service agent for Acme Corp."
        " Keep answers brief and conversational."
    ),
)
print(summary)
```

## Quick Start

### 1. Install the extra

```console
pip install "voip[mcp]"
```

### 2. Configure and run

```python
from voip.mcp import VoIPServer

server = VoIPServer(
    aor="sips:alice:secret@sip.example.com",
    # stun_server="stun.cloudflare.com:3478",  # default
    # verify_tls=True,                         # default
)
server.run()  # defaults to stdio transport
```

The server registers with the SIP carrier using the supplied Address of Record
(AOR), then listens for tool calls from the MCP client.

### Transport options

Pass `transport=` to \[`run()`\][voip.mcp.VoIPServer.run] to choose the MCP
wire protocol:

| Value               | Description                  |
| ------------------- | ---------------------------- |
| `"stdio"`           | Standard I/O (default)       |
| `"sse"`             | Server-Sent Events over HTTP |
| `"streamable-http"` | Streamable HTTP              |

## API Reference

::: voip.mcp
    options:
      members:
        - VoIPServer
        - MCPAgentCall
        - HangupDialog
        - OutboundMessageProtocol
        - MCPCallProtocol
