from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.schema import AIMessage, HumanMessage

app = FastAPI()

# Session store
sessions: Dict[str, Dict[str, any]] = {}

# --- Models ---
class ChatRequest(BaseModel):
    sessionid: Optional[str]
    message: str

class ChatResponse(BaseModel):
    sessionid: str
    response: str
    history: List[Dict[str, str]]

def create_agent_with_memory():
    llm = Ollama(base_url="http://localhost:11434", model="llama3.2:3b")
    
    tools = [
        Tool(name="Weather", func=weather_tool, description="Get weather info"),
        Tool(name="Cricket", func=cricket_tool, description="Get cricket score"),
        Tool(name="News", func=news_tool, description="Get top 10 news"),
        Tool(name="Flights", func=flights_tool, description="Get flight details"),
    ]
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
    )
    return agent, memory

def get_or_create_session(sessionid: Optional[str]) -> str:
    if sessionid and sessionid in sessions:
        return sessionid
    
    new_id = str(uuid.uuid4())
    agent, memory = create_agent_with_memory()
    
    sessions[new_id] = {
        "history": [],
        "agent": agent,
        "memory": memory
    }
    return new_id

# --- Tool functions ---
def weather_tool(_):
    return str({"location": "Delhi", "forecast": "Sunny", "temperature": "35C"})

def cricket_tool(_):
    return str({"match": "India vs Australia", "score": "250/3", "status": "India batting"})

def news_tool(_):
    return str({"top_10_news": [f"News {i}" for i in range(1, 11)]})

def flights_tool(_):
    return str({"flight": "AI202", "status": "On Time", "departure": "Delhi", "arrival": "Mumbai"})

# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    sessionid = get_or_create_session(request.sessionid)
    session_data = sessions[sessionid]
    history = session_data["history"]
    agent = session_data["agent"]
    memory = session_data["memory"]
    
    # Add to history before processing
    history.append({"user": request.message})
    
    try:
        # Manually add to memory
        memory.chat_memory.add_user_message(request.message)
        
        # Get response
        bot_reply = agent.run(input=request.message)
        
        # Add to memory
        memory.chat_memory.add_ai_message(bot_reply)
    except Exception as e:
        bot_reply = f"Sorry, I encountered an error: {str(e)}"
    
    history.append({"bot": bot_reply})
    
    return ChatResponse(
        sessionid=sessionid,
        response=bot_reply,
        history=history
    )

@app.get("/weather")
def get_weather():
    return weather_tool(None)

@app.get("/cricket")
def get_cricket():
    return cricket_tool(None)

@app.get("/news")
def get_news():
    return news_tool(None)

@app.get("/flights")
def get_flights():
    return flights_tool(None)