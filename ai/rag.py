from ai.embedder import embedder
from database.qdrant_db import qdrant_db
import time

def save_message_to_memory(user_id: str, text: str, role: str):
    """
    Saves a single message to Qdrant memory.
    role: "user" or "bot"
    """
    embedding = embedder.get_embedding(text)
    timestamp = time.time()
    qdrant_db.add_message(user_id, text, embedding, timestamp, role)

def get_memory_context(user_id: str, query: str, limit: int = 5) -> str:
    """
    Retrieves past messages related to the query for a given user.
    """
    query_embedding = embedder.get_embedding(query)
    results = qdrant_db.search_similar(user_id, query_embedding, limit=limit)
    
    if not results:
        return ""
        
    # Sort results by timestamp to maintain chronological order
    # (Qdrant returns them sorted by similarity, but we need time for context)
    sorted_results = sorted(results, key=lambda r: r.payload.get("timestamp", 0))
    
    context_lines = []
    for r in sorted_results:
        role = r.payload.get("role", "unknown")
        text = r.payload.get("text", "")
        # convert timestamp to human readable date/time? No, we just need the text flow.
        if role == "user":
            context_lines.append(f"Friend: {text}")
        else:
            context_lines.append(f"You: {text}")
            
    return "\n".join(context_lines)
