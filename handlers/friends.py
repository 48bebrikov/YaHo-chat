from telethon import events
from ai.gemini_engine import generate_reply
from database.sqlite_db import get_db_session, UserMetadata
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def register_friend_handlers(client, friends_list: list[str]):
    if not friends_list or friends_list == [""]:
        logger.warning("No friends listed in configuration. Bot will ignore everyone.")
        return

    @client.on(events.NewMessage(incoming=True, func=lambda e: e.is_private))
    async def handler(event):
        sender = await event.get_sender()
        sender_id = str(sender.id)
        sender_username = sender.username or ""

        # Check if the sender is in our friends list (by ID or username)
        if sender_id not in friends_list and sender_username not in friends_list:
            return

        user_id = sender_id # consistently use ID for Qdrant storage
        text = event.text
        if not text:
            # We skip pure media for now, though Gemini supports multimodal
            return

        logger.info(f"Received message from friend {user_id}: {text[:30]}...")

        try:
            # Generate reply
            reply_text = generate_reply(user_id, text)
            
            # Send the reply
            await event.reply(reply_text)
            
            # Update user metadata in SQLite
            db = get_db_session()
            try:
                user_meta = db.query(UserMetadata).filter(UserMetadata.user_id == user_id).first()
                if not user_meta:
                    user_meta = UserMetadata(user_id=user_id)
                    db.add(user_meta)
                
                user_meta.last_message_date = datetime.utcnow()
                # next_check_date can be set here if we want to reset proactive timer
                db.commit()
            except Exception as e:
                db.rollback()
                logger.error(f"DB Error: {e}")
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error generating reply: {e}")
