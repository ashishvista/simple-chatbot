from typing import Annotated, Union, List, TypedDict, Sequence
import operator
import os
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import StateGraph, END
from langchain_ollama.chat_models import ChatOllama
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_react_agent
from tools import tools
import langchain
from dotenv import load_dotenv
from langchain import hub
from langgraph.graph import StateGraph, START

load_dotenv()
langchain.debug = True
# Create a tool lookup dictionary for easy access
tool_lookup = {tool.name: tool for tool in tools}
# Get model names from environment variables
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]

# llm = ChatOllama(
#     base_url="http://localhost:11434",
#     temperature=0,
#     model=LLM_MODEL,
#     # system="You are a helpful assistant that can use tools to answer questions about finance, weather, sports, and travel.",
# )
# prompt = hub.pull("hwchase17/react")

# agent_runnable = create_react_agent(
#     llm,
#     tools,
#     prompt
# )

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url="http://localhost:11434/v1",
    api_key="unused",
    temperature=0.6
)



model=llm.bind_tools(tools)


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
