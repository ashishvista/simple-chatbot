from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="qwen3:4b")
persist_directory = "rapipay_loan_db"

def initialize_vector_db():
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
        # Load documents
        loader = DirectoryLoader('docs/', glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # Create and persist vector store
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectordb.persist()
        logger.info(f"Initialized vector DB with {len(texts)} chunks")
    else:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectordb

# Initialize vector DB at startup
vectordb = initialize_vector_db()

# Rapipay loan tool that uses vector DB
def rapipay_loan_tool(query: str) -> str:
    try:
        # Search similar documents
        docs = vectordb.similarity_search(query, k=3)
        
        # Format results
        result = ["Here's what I found about Rapipay loans:"]
        for i, doc in enumerate(docs, 1):
            result.append(f"\nDocument {i}:\n{doc.page_content}\n")
        
        return "\n".join(result) if len(result) > 1 else "No information found about Rapipay loans"
    except Exception as e:
        logger.error(f"Error in rapipay_loan_tool: {str(e)}")
        return f"Error searching loan info: {str(e)}"

def create_agent_with_memory():
    llm = Ollama(base_url="http://localhost:11434", model="qwen3:4b")
    
    tools = [
        Tool(name="Weather", func=weather_tool, description="Get weather info"),
        Tool(name="Cricket", func=cricket_tool, description="Get cricket score"),
        Tool(name="News", func=news_tool, description="Get top 10 news"),
        Tool(name="Flights", func=flights_tool, description="Get flight details"),
        Tool(
            name="RapipayLoans",
            func=rapipay_loan_tool,
            description="Useful for questions about loan interest rates, EMIs, or loan products from Rapipay"
        )
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
        sessions[sessionid]["last_accessed"] = datetime.now()
        return sessionid
    
    new_id = str(uuid.uuid4())
    agent, memory = create_agent_with_memory()
    
    sessions[new_id] = {
        "history": [],
        "agent": agent,
        "memory": memory,
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
    try:
        sessionid = get_or_create_session(request.sessionid)
        session_data = sessions[sessionid]
        history = session_data["history"]
        agent = session_data["agent"]
        memory = session_data["memory"]
        
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        history.append({"user": request.message})
        
        try:
            memory.chat_memory.add_user_message(request.message)
            bot_reply = agent.run(input=request.message)
            memory.chat_memory.add_ai_message(bot_reply)
        except Exception as e:
            logger.error(f"Agent error: {str(e)}", exc_info=True)
            bot_reply = "Sorry, I'm having trouble processing your request. Please try again."
        
        history.append({"bot": bot_reply})
        cleanup_old_sessions()
        
        return ChatResponse(
            sessionid=sessionid,
            response=bot_reply,
            history=history
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

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