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

# Register tools via MCP decorators with proper descriptions and schemas
@mcp.tool(description="Useful for questions about loan interest rates, EMIs, or loan products from Rapipay. Don't call it to generate random numbers")
def rapipay_loan_tool(query: str) -> str:
    """
    Args:
        query (str): Question about loan interest rates, EMIs, or loan products
    """
    return rapipay_loan_impl({"query": query})

@mcp.tool(description="Get weather info for a city")
def weather_tool(city: str) -> str:
    """
    Args:
        city (str): The name of the city to get weather information for
    """
    return weather_impl({"city": city})

@mcp.tool(description="Get live cricket match scores between two cricket teams ONLY. This tool is strictly for cricket matches, not for any other sport such as badminton, football, etc.")
def cricket_tool(team1: str, team2: str) -> str:
    """
    Args:
        team1 (str): Name of the first cricket team
        team2 (str): Name of the second cricket team
    """
    return cricket_impl({"team1": team1, "team2": team2})

@mcp.tool(description="Get top 10 news for a country")
def news_tool(country: str) -> str:
    """
    Args:
        country (str): Country name to get news for
    """
    return news_impl({"country": country})

@mcp.tool(description="Get flight details between two cities")
def flights_tool(source: str, destination: str) -> str:
    """
    Args:
        source (str): Source city name
        destination (str): Destination city name
    """
    return flights_impl({"source": source, "destination": destination})

@mcp.tool(description="Call this tool ONLY if none of the other tools are relevant to the user's request. When called, use your own knowledge to answer the prompt as best as possible.")
def fallback_tool(query: str) -> str:
    """
    Args:
        query (str): User's query that doesn't match other tools
    """
    return fallback_impl({"query": query})

@mcp.tool(description="Call this tool to format final LLM output. It extracts the final answer from LLM responses.")
def format_output_tool(content: str) -> str:
    """
    Args:
        content (str): Final output from LLM
    """
    if "</think>" in content:
        answer = content.split("</think>", 1)[1].strip()
        return answer
    return content

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