import logging
import os
import base64

from config import OPENROUTER_API_KEY, RECENT_CHAT_WINDOW_MESSAGES, OPENROUTER_MODEL_ID
from ai.rag import format_recent_chat_block, get_memory_context, persist_conversation_turn
from i18n import get_copy

logger = logging.getLogger(__name__)

def _encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def generate_reply(user_id: str, message: str, media_path: str = None) -> str:
    """Generates a reply taking into account RAG memory and optional media."""
    import asyncio
    
    recent_block = await asyncio.to_thread(
        format_recent_chat_block, user_id, message, RECENT_CHAT_WINDOW_MESSAGES
    )
    rag_context = await asyncio.to_thread(
        get_memory_context, user_id, message, limit_facts=5, limit_dialogue=5
    )

    copy = get_copy()
    if rag_context:
        prompt_text = (
            f"{recent_block}\n\n"
            f"{copy.rag_memory_intro}"
            f"{rag_context}\n\n"
            f"{copy.reply_naturally}"
        )
    else:
        prompt_text = (
            f"{recent_block}\n\n"
            f"{copy.reply_naturally}"
        )

    import datetime
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    current_time = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    prompt_content = []
    prompt_content.append({"type": "text", "text": f"[System Note: Current Date and Time is {current_time}]"})
    prompt_content.append({"type": "text", "text": prompt_text})

    if media_path and os.path.exists(media_path):
        try:
            base64_img = _encode_image_to_base64(media_path)
            prompt_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}"
                }
            })
        except Exception as e:
            logger.error(f"Failed to encode image {media_path}: {e}")

    from ai.graph_agent import run_react_agent
    
    # We pass the system prompt and the formatted prompt block to the ReAct agent
    reply_text = await run_react_agent(copy.system_prompt, prompt_content, user_id)

    now_save = datetime.datetime.now(datetime.timezone.utc)
    import asyncio
    asyncio.create_task(
        asyncio.to_thread(persist_conversation_turn,
            user_id,
            message,
            reply_text,
            sort_ts=now_save.timestamp(),
            event_utc_iso=now_save.isoformat(),
        )
    )

    from database.sqlite_db import append_friend_chat_turn
    asyncio.create_task(
        asyncio.to_thread(append_friend_chat_turn, user_id, message, reply_text)
    )
    
    # Trigger async Memory Graph execution to extract long-term facts
    from ai.memory_graph import run_memory_extraction_bg
    asyncio.create_task(run_memory_extraction_bg(user_id, message, reply_text))

    return reply_text
