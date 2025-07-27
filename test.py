import operator
from datetime import datetime
from typing import Annotated, TypedDict, Union

from dotenv import load_dotenv
from langchain import hub
import langchain
from langchain.agents import create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

load_dotenv()
langchain.debug = True

@tool
def get_now(format: str = "%Y-%m-%d %H:%M:%S"):
    """
    Get the current time
    """
    return datetime.now().strftime(format)

tools = [get_now]
# Create a tool lookup dictionary for easy access
tool_lookup = {tool.name: tool for tool in tools}

class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

model = ChatOllama(model="qwen3:4b")
prompt = hub.pull("hwchase17/react")
agent_runnable = create_react_agent(model, tools, prompt)

def execute_tools(state):
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

def run_agent(state):
    agent_outcome = agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}

def should_continue(state):
    messages = [state["agent_outcome"]]
    last_message = messages[-1]
    if "Action" not in last_message.log:
        return "end"
    else:
        return "continue"
    
workflow = StateGraph(AgentState)

workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "action", "end": END}
)

workflow.add_edge("action", "agent")
app = workflow.compile()

input_text = "Whats the current time?"

inputs = {"input": input_text, "chat_history": []}
results = []
for s in app.stream(inputs):
    result = list(s.values())[0]
    results.append(result)
    # print("----start------")
    # print(result)
    # print("---end------")