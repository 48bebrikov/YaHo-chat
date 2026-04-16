import asyncio
import logging
import json
from datetime import datetime
from database.sqlite_db import get_db_session, UserMetadata
from config import POLL_INTERVAL_SECONDS, FRIENDS_LIST
from ai.gemini_engine import get_gemini_model
from ai.rag import get_memory_context, save_message_to_memory
from ai.tools import search_saved_news

logger = logging.getLogger(__name__)

PROACTIVE_PROMPT = """You are deciding whether to proactively text a friend on Telegram.
You are an AI emulating a human friend, remember? 
Here is the context:
- User ID: {user_id}
- Hours since last interaction: {hours_since}
- Last topic: {last_topic}
- Recent news in channels: {news}

Consider:
1. Don't spam. If it's been less than a few hours, probably don't text unless there's breaking news.
2. If it's been a day or more, maybe just say hi or share a news piece.
3. Be natural.

You must respond in valid JSON format ONLY, with this structure:
{{
    "should_message": true or false,
    "message_text": "text to send if true, or empty string if false",
    "next_check_hours": integer (how many hours to wait before checking again if false, usually 1 to 24)
}}
"""

async def proactive_loop(client):
    logger.info(f"Starting proactive loop, interval {POLL_INTERVAL_SECONDS}s")
    while True:
        try:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            await check_and_message_friends(client)
        except Exception as e:
            logger.error(f"Error in proactive loop: {e}")

async def check_and_message_friends(client):
    if not FRIENDS_LIST or FRIENDS_LIST == [""]:
        return

    db = get_db_session()
    model = get_gemini_model()
    try:
        recent_news = search_saved_news("latest")
        
        for friend_id in FRIENDS_LIST:
            user_meta = db.query(UserMetadata).filter(UserMetadata.user_id == friend_id).first()
            
            hours_since = 24 # default if never interacted
            last_topic = "None"
            if user_meta:
                if user_meta.next_check_date and datetime.utcnow() < user_meta.next_check_date:
                    continue # Skip this user, it's not time yet
                
                delta = datetime.utcnow() - user_meta.last_message_date
                hours_since = round(delta.total_seconds() / 3600, 1)
                last_topic = user_meta.last_topic or "Unknown"

            # Get some RAG context to know the vibe
            context = get_memory_context(friend_id, "latest conversation", limit=3)

            prompt = PROACTIVE_PROMPT.format(
                user_id=friend_id,
                hours_since=hours_since,
                last_topic=last_topic,
                news=recent_news
            )
            
            if context:
                prompt += f"\n\nRecent conversation context:\n{context}"

            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            try:
                decision = json.loads(response.text)
                should_message = decision.get("should_message", False)
                message_text = decision.get("message_text", "")
                next_check_hours = decision.get("next_check_hours", 4)

                if not user_meta:
                    user_meta = UserMetadata(user_id=friend_id)
                    db.add(user_meta)

                if should_message and message_text:
                    # Send message
                    # Assuming friend_id is numeric string, otherwise we need to resolve username
                    try:
                        target = int(friend_id)
                    except ValueError:
                        target = friend_id # username
                    
                    logger.info(f"Proactively sending message to {target}: {message_text}")
                    await client.send_message(target, message_text)
                    
                    save_message_to_memory(friend_id, message_text, role="bot")
                    
                    user_meta.last_message_date = datetime.utcnow()
                    user_meta.next_check_date = None # reset
                else:
                    logger.info(f"Decided not to message {friend_id}. Checking again in {next_check_hours} hours.")
                    from datetime import timedelta
                    user_meta.next_check_date = datetime.utcnow() + timedelta(hours=next_check_hours)

                db.commit()

            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from Gemini: {response.text}")

    except Exception as e:
        logger.error(f"Failed during check_and_message_friends: {e}")
        db.rollback()
    finally:
        db.close()
