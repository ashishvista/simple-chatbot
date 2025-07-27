from typing import Annotated, Union, List, TypedDict
import operator
import os
import logging
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import StateGraph, END
from langchain_ollama.chat_models import ChatOllama
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_react_agent
from tools import tools
import langchain
from dotenv import load_dotenv
from langchain import hub


load_dotenv()
langchain.debug = True
# Create a tool lookup dictionary for easy access
tool_lookup = {tool.name: tool for tool in tools}
# Get model names from environment variables
LLM_MODEL = os.getenv("LLM_MODEL")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]

llm = ChatOllama(
    base_url="http://localhost:11434",
    temperature=0,
    model=LLM_MODEL,
    # system="You are a helpful assistant that can use tools to answer questions about finance, weather, sports, and travel.",
)
prompt = hub.pull("hwchase17/react")

agent_runnable = create_react_agent(
    llm,
    tools,
    prompt
)

def run_agent(state: AgentState):
    agent_outcome = agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}
def execute_tools(state: AgentState):
 # print("Called `execute_tools`")
    agent_action = state["agent_outcome"]
    
    if not isinstance(agent_action, AgentAction):
        raise ValueError("Expected AgentAction")
    
    tool_name = agent_action.tool
    # print(f"Calling tool: {tool_name}")

    # Get the tool from our tools dictionary
    selected_tool = tool_lookup[tool_name]
    # Call the tool directly with the tool input
    response = selected_tool.invoke(agent_action.tool_input)
    return {"intermediate_steps": [(agent_action, response)]}

def should_continue(state: AgentState):
    action = state["agent_outcome"]
    return "continue" if isinstance(action, AgentAction) else "end"

workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
workflow.add_edge("action", "agent")
agent_workflow = workflow.compile()
