import logging
import asyncio
from typing import TypedDict, Annotated, Sequence, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import json

from config import GEMINI_API_KEY, GEMINI_MODEL_ID
from ai.rag import qdrant_client

logger = logging.getLogger(__name__)

# --- State ---
class MemoryState(TypedDict):
    user_id: str
    user_message: str
    bot_reply: str
    extracted_facts: List[str]
    topic: str
    
# --- LLM ---
def get_memory_llm():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_ID,
        api_key=GEMINI_API_KEY,
        temperature=0.1,
    )

# --- Nodes ---
def analyze_dialogue(state: MemoryState):
    """Analyzes the latest turn to extract facts and current topic."""
    llm = get_memory_llm()
    
    prompt = f"""
    Analyze the following dialogue turn between a user and an AI assistant named Katya.
    Extract key facts about the user (preferences, job, relationships, plans, mood).
    Return a JSON object with:
    1. "facts": a list of string facts (if any, otherwise empty list)
    2. "topic": a short string describing what they are talking about right now.
    
    Dialogue:
    User: {state['user_message']}
    Katya: {state['bot_reply']}
    
    Output strictly as JSON.
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Parse JSON
        text = response.content
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            
        data = json.loads(text)
        return {
            "extracted_facts": data.get("facts", []),
            "topic": data.get("topic", "general chat")
        }
    except Exception as e:
        logger.error(f"Error in memory extraction: {e}")
        return {"extracted_facts": [], "topic": "general chat"}


def update_database(state: MemoryState):
    """Saves the extracted facts into Qdrant and updates UserMetadata."""
    facts = state.get("extracted_facts", [])
    user_id = state.get("user_id")
    topic = state.get("topic")
    
    # 1. Update UserMetadata with the current topic
    try:
        from database.sqlite_db import db_session, UserMetadata
        with db_session() as db:
            user_meta = db.query(UserMetadata).filter(UserMetadata.user_id == user_id).first()
            if user_meta:
                user_meta.last_topic = topic
    except Exception as e:
        logger.error(f"Failed to update user topic in sqlite: {e}")

    # 2. Save facts to Qdrant (Knowledge Graph approximation)
    if facts and qdrant_client:
        import uuid
        from fastembed import TextEmbedding
        try:
            # Reusing the model from ai.rag if possible, or initializing a simple one.
            # In YaHo we just used generic qdrant points. We will use the existing `ai.rag.embed_model`
            from ai.rag import embed_model
            
            for fact in facts:
                # Add "User: " prefix to make it clear this is about the user
                text = f"Fact about user: {fact}"
                embs = list(embed_model.embed([text]))
                vector = embs[0].tolist()
                
                point_id = str(uuid.uuid4())
                
                qdrant_client.upsert(
                    collection_name=f"chat_{user_id}",
                    points=[{
                        "id": point_id,
                        "vector": vector,
                        "payload": {
                            "text": text,
                            "type": "fact",
                            "timestamp": __import__("datetime").datetime.now().timestamp()
                        }
                    }]
                )
                logger.info(f"Memory Graph: Saved fact for {user_id}: {fact}")
        except Exception as e:
            logger.error(f"Failed to insert facts to Qdrant: {e}")
            
    return {}

# --- Build Graph ---
def build_memory_graph():
    workflow = StateGraph(MemoryState)
    
    workflow.add_node("analyze", analyze_dialogue)
    workflow.add_node("save", update_database)
    
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "save")
    workflow.add_edge("save", END)
    
    return workflow.compile()

memory_graph = build_memory_graph()

async def run_memory_extraction_bg(user_id: str, user_message: str, bot_reply: str):
    """Triggers the memory graph in the background."""
    state = {
        "user_id": user_id,
        "user_message": user_message,
        "bot_reply": bot_reply,
        "extracted_facts": [],
        "topic": ""
    }
    
    try:
        await memory_graph.ainvoke(state)
    except Exception as e:
        logger.error(f"Memory graph failed: {e}")
