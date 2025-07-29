from typing import Annotated, TypedDict, Sequence
import os
from langchain_core.messages import SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from tools import tools
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
import httpx
import requests
from mcp_client import call_mcp_tool
import langchain
load_dotenv()
# Create a tool lookup dictionary for easy access
tool_lookup = {tool.name: tool for tool in tools}
# Get model names from environment variables
LLM_MODEL = os.getenv("LLM_MODEL")


class LoggingHTTPTransport(httpx.HTTPTransport):
    def handle_request(self, request):
        print(f"HTTPX REQUEST: {request.method} {request.url}")
        if request.content:
            print(f"HTTPX REQUEST BODY: {request.content}")
        response = super().handle_request(request)
        print(f"HTTPX RESPONSE STATUS: {response.status_code}")
        print(f"HTTPX RESPONSE BODY: {response.read()}")
        response._content = response.content  # Reset content for downstream consumers
        return response

# Create a custom httpx.Client with the logging transport
httpx_client = httpx.Client(transport=LoggingHTTPTransport())
langchain.debug = os.getenv("LANGCHAIN_DEBUG", "false").lower() == "true"

from langchain_openai import ChatOpenAI

class LLMConfig:
    def __init__(self, use_httpx_client: bool = False):
        self.model = LLM_MODEL
        self.base_url = os.getenv("OLLAMA_SERVER_URL")
        self.api_key = "unused"
        self.temperature = 0.6
        self.http_client = httpx_client if use_httpx_client else None

# Set this flag from environment variable (default: True)
USE_HTTPX_CLIENT_LOGGING = os.getenv("USE_HTTPX_CLIENT_LOGGING").lower() == "true"
llm_config = LLMConfig(use_httpx_client=USE_HTTPX_CLIENT_LOGGING)

llm_kwargs = {
    "model": llm_config.model,
    "base_url": llm_config.base_url,
    "api_key": llm_config.api_key,
    "temperature": llm_config.temperature,
}
if llm_config.http_client:
    llm_kwargs["http_client"] = llm_config.http_client

llm = ChatOpenAI(**llm_kwargs)


model = llm.bind_tools(tools)


class MessagesState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], ...]  # for state

def call_llm(state: MessagesState) -> MessagesState:
    # Explicitly enumerate the available tools in the system prompt
    tool_names = ", ".join([tool.name for tool in tools])
    system = SystemMessage(
        content=(
            "You are a helpful assistant. "
            f"The only tools you can use are: {tool_names}. "
            "Do not invent or mention any tool that is not in this list. "
            "If none of the tools are relevant, do not call any tool at all. Use your own knowledge to answer the question but dont say None of the tools are relevant. "
        )
    )
    response = model.invoke([system] + state["messages"])
    return {"messages": [*state["messages"], response]}


def call_tools(state: MessagesState) -> MessagesState:
    from langchain_core.messages import ToolMessage
    msgs = state["messages"]
    ai_msg = msgs[-1]
    results = []
    for tool_call in ai_msg.tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["args"]
        obs = call_mcp_tool(tool_name, args)
        results.append(ToolMessage(content=str(obs), tool_call_id=tool_call["id"]))
    return {"messages": [*msgs, *results]}

from langgraph.graph import END
def should_continue(state: MessagesState) -> str:
    return "Action" if state["messages"][-1].tool_calls else END

graph = StateGraph(MessagesState)
graph.add_node("llm", call_llm)
graph.add_node("action", call_tools)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, {"Action": "action", END: END})
graph.add_edge("action", "llm")

agent_workflow = graph.compile()


