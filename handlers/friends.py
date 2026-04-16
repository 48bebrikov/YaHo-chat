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
            # 1. Determine delay based on last interaction
            delay_seconds = random.randint(60, 600) # Default 5-10 minutes for inactive chats
            
            db = get_db_session()
            try:
                user_meta = db.query(UserMetadata).filter(UserMetadata.user_id == user_id).first()
                if user_meta and user_meta.last_message_date:
                    delta = datetime.utcnow() - user_meta.last_message_date
                    if delta.total_seconds() < 15 * 60: # If we chatted in the last 15 mins
                        delay_seconds = random.randint(5, 30) # Quick reply
            except Exception as e:
                logger.error(f"DB Error checking last message: {e}")
            finally:
                db.close()
                
            logger.info(f"Waiting {delay_seconds}s before reading message from {user_id}")
            await asyncio.sleep(delay_seconds)
            
            # Mark message as read
            await client.send_read_acknowledge(event.chat_id)
            
            # Generate reply in a separate thread so we don't block the async event loop
            reply_text = await asyncio.to_thread(generate_reply, user_id, text, media_path)
            
            # Split reply into multiple messages (by newlines or sentence boundaries)
            import re
            parts = [p.strip() for p in re.split(r'\n+', reply_text) if p.strip()]
            
            final_parts = []
            for p in parts:
                if len(p) > 40 and re.search(r'[.!?]\s', p):
                    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p) if s.strip()]
                    final_parts.extend(sentences)
                else:
                    final_parts.append(p)
            
            # Send messages one by one with typing simulation
            for i, part in enumerate(final_parts):
                typing_delay = min(len(part) / 15, 4)
                
                action = 'typing' if not media_path else 'document'
                async with client.action(event.chat_id, action):
                    await asyncio.sleep(typing_delay)
                    
                if i == 0:
                    await event.reply(part)
                else:
                    await event.respond(part)
                
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
