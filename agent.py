from typing import Annotated, TypedDict, Sequence
import os
from langchain_core.messages import SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from tools import tools
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
import httpx

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
# langchain.debug = True

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url="http://localhost:11434/v1",
    api_key="unused",
    temperature=0.6,
    http_client=httpx_client  # Pass the custom client for logging
)


model = llm.bind_tools(tools)


class MessagesState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], ...]  # for state

def call_llm(state: MessagesState) -> MessagesState:
    system = SystemMessage(content="You are a helpful assistant.")
    response = model.invoke([system] + state["messages"])
    return {"messages": [*state["messages"], response]}

def call_tools(state: MessagesState) -> MessagesState:
    # Extract tool calls from the last AI message
    from langchain_core.messages import ToolMessage
    msgs = state["messages"]
    ai_msg = msgs[-1]
    results = []
    for tool_call in ai_msg.tool_calls:
        tool_fn = tool_lookup.get(tool_call["name"])
        obs = tool_fn.invoke(tool_call["args"])
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


