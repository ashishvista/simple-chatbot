from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict, Optional
import requests
import uuid
from langchain.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

app = FastAPI()

# In-memory session store
sessions: Dict[str, List[Dict[str, str]]] = {}

# --- Models ---
class ChatRequest(BaseModel):
    sessionid: Optional[str]
    message: str

class ChatResponse(BaseModel):
    sessionid: str
    response: str
    history: List[Dict[str, str]]

# --- Helper functions ---
def get_or_create_session(sessionid: Optional[str]) -> str:
    if sessionid and sessionid in sessions:
        return sessionid
    new_id = str(uuid.uuid4())
    sessions[new_id] = []
    return new_id

# --- Hardcoded APIs ---
@app.get("/weather")
def get_weather():
    return {"location": "Delhi", "forecast": "Sunny", "temperature": "35C"}

@app.get("/cricket")
def get_cricket():
    return {"match": "India vs Australia", "score": "250/3", "status": "India batting"}

@app.get("/news")
def get_news():
    return {"top_10_news": [f"News {i}" for i in range(1, 11)]}

@app.get("/flights")
def get_flights():
    return {"flight": "AI202", "status": "On Time", "departure": "Delhi", "arrival": "Mumbai"}

# --- LangChain LLM Setup ---
ollama_llm = Ollama(base_url="http://localhost:11434", model="llama3.2:3b")
memory = ConversationBufferMemory()

# --- Tool functions for LangChain ---
def weather_tool(_):
    return str(get_weather())
def cricket_tool(_):
    return str(get_cricket())
def news_tool(_):
    return str(get_news())
def flights_tool(_):
    return str(get_flights())

# Register tools for agent
langchain_tools = [
    Tool(name="Weather", func=weather_tool, description="Get weather info"),
    Tool(name="Cricket", func=cricket_tool, description="Get cricket score"),
    Tool(name="News", func=news_tool, description="Get top 10 news"),
    Tool(name="Flights", func=flights_tool, description="Get flight details"),
]

agent = initialize_agent(
    langchain_tools,
    ollama_llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=False,
)

# --- Chatbot API ---
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    sessionid = get_or_create_session(request.sessionid)
    history = sessions[sessionid]
    history.append({"user": request.message})

    # Use LangChain agent to determine context and run tools
    memory.chat_memory.messages = []
    for msg in history:
        if "user" in msg:
            memory.chat_memory.add_user_message(msg["user"])
        if "bot" in msg:
            memory.chat_memory.add_ai_message(msg["bot"])
    try:
        bot_reply = agent.invoke(
            {
                "input": request.message,
                "chat_history": memory.buffer
            }
        )
    except Exception as e:
        bot_reply = f"Agent error: {e}"
    history.append({"bot": bot_reply})
    sessions[sessionid] = history
    return ChatResponse(sessionid=sessionid, response=bot_reply, history=history)

