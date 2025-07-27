from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(...)

@tool(args_schema=SearchInput)
def my_search(query: str):
    return f"Searching: {query}"

llm = ChatOllama(model="mixtral-8x7b", temperature=0)
tools = [my_search]
prompt = hub.pull("hwchase17/react-json").partial(
    tools=tools, tool_names=[tool.name for tool in tools]
)
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

executor.invoke({"input": "Find the capital of France"})