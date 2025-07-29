import os
import requests

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:9000/tools")

def call_mcp_tool(tool_name: str, args: dict) -> str:
    endpoint = f"{MCP_SERVER_URL}/{tool_name}"
    try:
        resp = requests.post(endpoint, json=args, timeout=10)
        resp.raise_for_status()
        return resp.json().get("result", "No result")
    except Exception as e:
        return f"Error calling MCP tool: {str(e)}"
