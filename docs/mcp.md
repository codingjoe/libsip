# MCP Server

The `voip` package ships a ready-made [Model Context Protocol (MCP)][mcp] server
that exposes tools to make phone calls on your behalf to any MCP client.

## Claude Code setup

Add the server to your MCP config (see [Claude Code MCP docs][cc-mcp]):

```json
{
  "mcpServers": {
    "voip": {
      "command": "voip",
      "args": [
        "mcp"
      ],
      "env": {
        "SIP_AOR": "sip:youruser@carrier.example"
      }
    }
  }
}
```

Set `SIP_AOR` to your SIP address-of-record. The transport (TLS vs TCP) and proxy
address are derived from the URI automatically.

## Tools

::: voip.mcp.say

::: voip.mcp.call

[cc-mcp]: https://docs.anthropic.com/en/docs/claude-code/mcp
[mcp]: https://modelcontextprotocol.io/
