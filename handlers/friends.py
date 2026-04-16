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
        
        # Check if it's media or text
        text = event.text or ""
        media_path = None
        
        if event.photo:
            # We skip downloading huge files, but let's download photos to send to Gemini
            logger.info("Received a photo, downloading...")
            media_path = await event.download_media(file="database/")
            if not text:
                text = "[Пользователь прислал фото]"

        if not text and not media_path:
            return

        logger.info(f"Received message from friend {user_id}: {text[:30]}...")

        import asyncio
        import random
        
        try:
            # Simulate the time it takes to notice a notification and open the app
            # (between 3 and 300 seconds). This runs asynchronously, so other people's 
            # messages are still processed in parallel!
            await asyncio.sleep(random.randint(10, 600))
            
            # Mark message as read
            await client.send_read_acknowledge(event.chat_id)
            
            # Tell telegram we are typing...
            action = 'typing' if not media_path else 'document'
            async with client.action(event.chat_id, action):
                # Generate reply in a separate thread so we don't block the async event loop
                # This ensures true parallelism for multiple friends talking at once
                reply_text = await asyncio.to_thread(generate_reply, user_id, text, media_path)
                
                # Add an artificial delay based on message length 
                # to make typing look more human (e.g. max 5 seconds)
                await asyncio.sleep(min(len(reply_text) / 15, 5))
                
                # Send the reply
                await event.reply(reply_text)
                
            # Cleanup downloaded media
            if media_path:
                import os
                try:
                    os.remove(media_path)
                except Exception as e:
                    logger.error(f"Failed to delete {media_path}: {e}")
            
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
