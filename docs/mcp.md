# MCP Server

The `voip` package ships a ready-made [Model Context Protocol (MCP)][MCP] server that exposes
two tools — `say` and `call` — so that any MCP-compatible AI client (e.g. Claude Desktop)
can make phone calls on your behalf.

[MCP]: https://modelcontextprotocol.io/

## Installation

```console
pip install voip[mcp]
```

## Starting the server

The server is available as a CLI command.  Pass the SIP address-of-record (AOR) either as a
positional argument or via the `SIP_AOR` environment variable:

```console
voip mcp sip:alice@provider.example --transport stdio
# or
SIP_AOR=sip:alice@provider.example voip mcp --transport http
```

| Variable            | Default                    | Description                                      |
|---------------------|----------------------------|--------------------------------------------------|
| `SIP_AOR`           | *(required)*               | SIP address-of-record for registration.          |
| `STUN_SERVER`       | `stun.cloudflare.com:3478` | STUN server used for RTP NAT traversal.          |
| `SIP_NO_VERIFY_TLS` | unset                      | Set to `1` to disable TLS certificate checks.   |

## Tools

::: voip.mcp.say

::: voip.mcp.call

## Transport factory functions

The two factory functions below are the public building blocks used by the MCP tools.
They follow the same **start-and-block** pattern as
[`mcp.run`][fastmcp.FastMCP.run]: call them with the desired parameters and they handle
the full connection lifecycle internally.

::: voip.mcp.connect_rtp

::: voip.mcp.connect_sip

## Environment variable helpers

::: voip.mcp.read_aor

::: voip.mcp.read_stun_server

## Session classes

::: voip.mcp.HangupDialog

::: voip.mcp.MCPAgentCall
