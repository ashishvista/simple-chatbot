from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, TypedDict, Annotated, Sequence, Union
import uuid
import operator
from datetime import datetime, timedelta
import os
import logging
from fastapi.middleware.cors import CORSMiddleware

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_ollama.chat_models import ChatOllama
from langchain import hub

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class ChatRequest(BaseModel):
    sessionid: Optional[str]
    message: str

class ChatResponse(BaseModel):
    sessionid: str
    response: str
    history: List[Dict[str, str]]

# --- State Definition for LangGraph ---
class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

# --- Embeddings and Vector DB ---
embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="qwen3:4b")
persist_directory = "rapipay_loan_db"

def initialize_vector_db():
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
        loader = DirectoryLoader('docs/', glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        logger.info(f"Initialized vector DB with {len(texts)} chunks")
    else:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectordb

vectordb = initialize_vector_db()

# --- Tool Definitions (existing tools) ---
@tool
def rapipay_loan_tool(query: str) -> str:
    """Useful for questions about loan interest rates, EMIs, or loan products from Rapipay"""
    try:
        docs = vectordb.similarity_search(query, k=3)
        result = ["Here's what I found about Rapipay loans:"]
        for i, doc in enumerate(docs, 1):
            result.append(f"\nDocument {i}:\n{doc.page_content}\n")
        return "\n".join(result) if len(result) > 1 else "No information found about Rapipay loans"
    except Exception as e:
        logger.error(f"Error in rapipay_loan_tool: {str(e)}")
        return f"Error searching loan info: {str(e)}"

@tool
def weather_tool(city: str) -> str:
    """Get weather info for a city.
    
    Args:
        city (str): The name of the city to get weather information for.
    """
    if not city:
        raise ValueError("City is required for weather info.")
    return str({"location": city, "forecast": "Sunny", "temperature": "35C"})

@tool
def cricket_tool(team1: str, team2: str) -> str:
    """Get cricket match scores."""
    if not team1 or not team2:
        raise ValueError("Both team names are required for cricket score.")
    return str({"match": f"{team1} vs {team2}", "score": "250/3", "status": f"{team1} batting"})

@tool
def news_tool(country: str) -> str:
    """Get top 10 news for a country."""
    if not country:
        raise ValueError("Country is required for news.")
    return str({"country": country, "top_10_news": [f"News {i}" for i in range(1, 11)]})

@tool
def flights_tool(source: str, destination: str) -> str:
    """Get flight details between two cities."""
    if not source or not destination:
        raise ValueError("Source and destination cities are required for flight details.")
    return str({"flight": "AI202", "status": "On Time", "departure": source, "arrival": destination})

# --- Tool List and Executor ---
tools = [rapipay_loan_tool, weather_tool, cricket_tool, news_tool, flights_tool]

# --- LLM and Agent Setup ---
llm = ChatOllama(
    base_url="http://localhost:11434",
    temperature=0,
    model="qwen3:4b",
    system="You are a helpful assistant that can use tools to answer questions about finance, weather, sports, and travel."
)
agent_runnable = create_react_agent(
    llm,
    tools=tools,
    prompt=hub.pull("hwchase17/react")
)

def run_agent(state: AgentState):
    outcome = agent_runnable.invoke(state)
    return {"agent_outcome": outcome}

def execute_tool(state: AgentState):
    action = state["agent_outcome"]
    tool_map = {tool.name: tool for tool in tools}
    result = tool_map[action.tool].invoke(action.tool_input)
    return {"intermediate_steps": [(action, result)]}

def should_continue(state: AgentState):
    action = state["agent_outcome"]
    return "continue" if isinstance(action, AgentAction) else END

# --- LangGraph Workflow ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
workflow.add_edge("action", "agent")
compiled_workflow = workflow.compile()

# --- Session Management ---
sessions: Dict[str, Dict[str, any]] = {}

def get_or_create_session(sessionid: Optional[str]) -> str:
    if sessionid and sessionid in sessions:
        sessions[sessionid]["last_accessed"] = datetime.now()
        return sessionid
    new_id = str(uuid.uuid4())
    sessions[new_id] = {
        "history": [],
        "last_accessed": datetime.now()
    }
    return new_id

def cleanup_old_sessions(hours=2):
    now = datetime.now()
    expired = [sid for sid, data in sessions.items()
              if now - data["last_accessed"] > timedelta(hours=hours)]
    for sid in expired:
        del sessions[sid]
    logger.info(f"Cleaned up {len(expired)} expired sessions")

# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        sessionid = get_or_create_session(request.sessionid)
        session_data = sessions[sessionid]
        history = session_data["history"]

        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        history.append({"user": request.message})

        try:
            initial_state = {
                "input": request.message,
                "chat_history": [],
                "agent_outcome": None,
                "intermediate_steps": []
            }
            result = await compiled_workflow.ainvoke(initial_state)
            # Find the final answer from agent_outcome or intermediate_steps
            reply_text = ""
            if result.get("agent_outcome") and isinstance(result["agent_outcome"], AgentFinish):
                reply_text = result["agent_outcome"].return_values.get("output", "")
            elif result.get("intermediate_steps"):
                # If the last step is a tool result, show that
                reply_text = str(result["intermediate_steps"][-1][1])
            else:
                reply_text = "Sorry, I couldn't process your request."
            history.append({"bot": reply_text})
        except Exception as e:
            logger.error(f"Agent error: {str(e)}", exc_info=True)
            reply_text = "Sorry, I'm having trouble processing your request. Please try again."

        cleanup_old_sessions()

        return ChatResponse(
            sessionid=sessionid,
            response=reply_text,
            history=history
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/weather")
def get_weather():
    return weather_tool.invoke({"city": "London"})

@app.get("/cricket")
def get_cricket():
    return cricket_tool.invoke({"team1": "India", "team2": "Australia"})

@app.get("/news")
def get_news():
    return news_tool.invoke({"country": "India"})

@app.get("/flights")
def get_flights():
    return flights_tool.invoke({"source": "Delhi", "destination": "Mumbai"})

# Explanation of agent and tool_node in the LangGraph workflow:
#
# - agent node:
#     This node sends the current conversation (messages) to the LLM (ChatOllama).
#     The LLM generates a response, which may be a direct answer or a tool call (e.g., {"tool": "weather_tool", "input": {"city": "Delhi"}}).
#     If the response is a tool call, the workflow routes to the tool_node.
#     If not, the workflow ends and the response is returned to the user.
#
# - tool_node:
#     This node parses the tool call(s) from the LLM's output.
#     It invokes the appropriate Python function (tool) with the arguments provided by the LLM.
#     The result(s) from the tool(s) are formatted as messages and passed back to the agent node.
#
# Example:
#   User: "What's the weather in Delhi?"
#   1. agent node: LLM receives the message and responds with a tool call:
#      {"tool": "weather_tool", "input": {"city": "Delhi"}}
#   2. tool_node: Parses this tool call, calls weather_tool(city="Delhi"), gets the weather info, and returns it as a message.
#   3. agent node: LLM receives the tool result and generates a final answer for the user.
#   4. If no further tool calls are needed, the workflow ends and the answer is returned.
#
# This loop allows the agent to use external tools/functions as needed to answer user queries.