from mcp.server.fastmcp import FastMCP
import json
import asyncio

# Alias imports to avoid recursion
from tools import (
    rapipay_loan_tool as rapipay_loan_impl,
    weather_tool as weather_impl,
    cricket_tool as cricket_impl,
    news_tool as news_impl,
    flights_tool as flights_impl,
    fallback_tool as fallback_impl,
)

mcp = FastMCP("lots of tools")

# Register tools via MCP decorators, call the aliased implementations
@mcp.tool()
def rapipay_loan_tool(params):
    return rapipay_loan_impl(params)

@mcp.tool()
def weather_tool(params):
    return weather_impl(params)

@mcp.tool()
def cricket_tool(params):
    return cricket_impl(params)

@mcp.tool()
def news_tool(params):
    return news_impl(params)

@mcp.tool()
def flights_tool(params):
    return flights_impl(params)

@mcp.tool()
def fallback_tool(params):
    return fallback_impl(params)

# Use MCP's streamable HTTP app
app = mcp.streamable_http_app

# Run server in appropriate mode
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--stdio":
        # MCP stdio mode for n8n and other MCP clients
        mcp.run()
    else:
        # HTTP mode for direct API access
        import uvicorn
        uvicorn.run("mcp_server:app", host="0.0.0.0", port=9000, reload=True)