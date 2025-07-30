from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
from datetime import datetime, timedelta
import logging
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from guardrails import Guard, OnFailAction
from guardrails.hub import ProfanityFree,  ToxicLanguage, CompetitorCheck, RegexMatch, BanList, ValidLength, DetectJailbreak


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

# Define the guard with both validators using explicit classes and exception on fail
guard = Guard().use(
    ProfanityFree(on_fail=OnFailAction.EXCEPTION),
).use_many(
    ToxicLanguage(threshold=0.5, on_fail=OnFailAction.EXCEPTION),
    # CompetitorCheck(["Apple", "Microsoft"], on_fail=OnFailAction.EXCEPTION),
    # RegexMatch(regex=r"\d{3}-\d{3}-\d{4}", on_fail=OnFailAction.EXCEPTION),
    # BanList(banned_words=["secret", "codename"], max_l_dist=1, on_fail=OnFailAction.EXCEPTION),
    # ValidLength(min=1, max=200, on_fail=OnFailAction.EXCEPTION),
    DetectJailbreak(on_fail=OnFailAction.EXCEPTION)
)

# --- Models ---
class ChatRequest(BaseModel):
    sessionid: Optional[str]
    message: str

class ChatResponse(BaseModel):
    sessionid: str
    response: str
    # history: List[Dict[str, str]]

# --- Session Management ---
sessions: Dict[str, Dict[str, any]] = {}

def get_or_create_session(sessionid: Optional[str]) -> str:
    if sessionid:
        if sessionid in sessions:
            sessions[sessionid]["last_accessed"] = datetime.now()
        else:
            # Create a new session with the provided sessionid
            sessions[sessionid] = {
                "history": [],
                "last_accessed": datetime.now()
            }
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
        # Guardrails AI check for user input
        try:
            guard.validate(request.message)
        except Exception:
            return ChatResponse(
                sessionid=request.sessionid or "",
                response="⚠️ Your message was blocked due to unsafe or inappropriate content.",
                history=[]
            )
        sessionid = get_or_create_session(request.sessionid)
        session_data = sessions[sessionid]
        history = session_data["history"]

        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        history.append({"user": request.message})

        try:
            # Build messages from history
            messages = []
            for entry in history:
                if "user" in entry:
                    messages.append(HumanMessage(content=entry["user"]))
                elif "bot" in entry:
                    messages.append(AIMessage(content=entry["bot"]))

            # Add the latest user message if not already present
            if not messages or not isinstance(messages[-1], HumanMessage):
                messages.append(HumanMessage(content=request.message))

            initial_state = {
                "messages": messages
            }

            state = agent_workflow.invoke(initial_state)
            for m in state["messages"]:
                m.pretty_print()

            reply_text = unwrap_final(state)
            # Guardrails AI check for chatbot response
            try:
                guard.validate(reply_text)
            except Exception:
                reply_text = "⚠️ The chatbot's response was blocked due to unsafe or inappropriate content."
            history.append({"bot": reply_text})
        except Exception as e:
            logger.error(f"Agent error: {str(e)}", exc_info=True)
            reply_text = "Sorry, I'm having trouble processing your request. Please try again."

        cleanup_old_sessions()

        return ChatResponse(
            sessionid=sessionid,
            response=reply_text,
            # history=history
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    
def unwrap_final(state):
    from langchain_core.messages import AIMessage
               # Extract the final AIMessage
    final_ai = next(
        (msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)), 
        None
    )
    content = final_ai.content
    if "</think>" in content:
        answer = content.split("</think>", 1)[1].strip()
    else:
        answer = content
    return answer