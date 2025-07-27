from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, TypedDict, Annotated, Sequence, Union
import uuid
from datetime import datetime, timedelta
import os
import logging
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_core.agents import AgentAction, AgentFinish

from agent import agent_workflow

# Load environment variables from .env file
load_dotenv()


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
            result = await agent_workflow.ainvoke(initial_state)
            # Find the final answer from agent_outcome or intermediate_steps
            reply_text = ""
            if result.get("agent_outcome") and isinstance(result["agent_outcome"], AgentFinish):
                reply_text = result["agent_outcome"].return_values.get("output", "")

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