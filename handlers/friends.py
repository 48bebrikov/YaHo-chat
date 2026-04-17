import asyncio
import logging
import os
import random
import re
from datetime import datetime

from telethon import events

from ai.gemini_engine import generate_reply
from config import (
    BUFFER_QUIET_SECONDS,
    FRIEND_REPLY_DELAY_COLD_MAX,
    FRIEND_REPLY_DELAY_COLD_MIN,
    FRIEND_REPLY_DELAY_WARM_MAX,
    FRIEND_REPLY_DELAY_WARM_MIN,
    FRIEND_REPLY_WARM_WINDOW_MINUTES,
    FRIEND_THINKING_AFTER_READ_MAX,
    FRIEND_THINKING_AFTER_READ_MIN,
    FRIEND_TYPING_PER_PART_MAX,
)
from database.sqlite_db import get_db_session, UserMetadata
logger = logging.getLogger(__name__)

# Global buffer to store incoming messages
message_buffers = {}


async def _wait_until_quiet_buffer(user_id: str, quiet: float) -> None:
    """Wait until the buffer stops growing for `quiet` seconds (user finished the burst)."""
    while True:
        buf = message_buffers.get(user_id)
        if not buf:
            return
        prev_count = len(buf)
        await asyncio.sleep(quiet)
        buf = message_buffers.get(user_id)
        if not buf:
            return
        if len(buf) == prev_count:
            return


def _record_user_message_activity(user_id: str) -> None:
    """Mark that the user just wrote in private — proactive loop uses this to avoid interrupting a live chat."""
    db = get_db_session()
    try:
        user_meta = db.query(UserMetadata).filter(UserMetadata.user_id == user_id).first()
        if not user_meta:
            user_meta = UserMetadata(user_id=user_id)
            db.add(user_meta)
        user_meta.last_user_message_at = datetime.utcnow()
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"DB Error recording user activity: {e}")
    finally:
        db.close()

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
        
        # Add to buffer to aggregate multiple fast messages
        if user_id not in message_buffers:
            message_buffers[user_id] = []
            
        text = event.text or ""
        media_path = None
        
        if event.photo:
            logger.info("Received a photo, downloading...")
            media_path = await event.download_media(file="database/")
            if not text:
                text = "[Пользователь прислал фото]"

        if not text and not media_path:
            return
            
        message_buffers[user_id].append({
            "text": text,
            "media_path": media_path,
            "event": event
        })
        _record_user_message_activity(user_id)

        # If there's already a pending processing task for this user, let it handle this new message too
        if getattr(client, f"_processing_{user_id}", False):
            return
            
        # Lock processing for this user
        setattr(client, f"_processing_{user_id}", True)
        
        # Start a background task to process the buffer after a short delay
        asyncio.create_task(process_buffered_messages(client, user_id))

async def process_buffered_messages(client, user_id: str):
    try:
        # Wait until the user stops sending for a few seconds (merges split thoughts into one reply)
        await _wait_until_quiet_buffer(user_id, BUFFER_QUIET_SECONDS)

        messages = message_buffers.pop(user_id, [])

        if not messages:
            return

        # Combine text from all messages
        combined_text = "\n".join([m["text"] for m in messages if m["text"]])
        
        # Grab the last media path if there are multiple
        media_paths = [m["media_path"] for m in messages if m["media_path"]]
        final_media_path = media_paths[-1] if media_paths else None
        
        # The event we will reply to (the last one)
        last_event = messages[-1]["event"]

        logger.info(f"Processing aggregated messages from {user_id}: {combined_text[:30]}...")

        try:
            # 1. Determine delay based on last interaction
            delay_seconds = random.randint(FRIEND_REPLY_DELAY_COLD_MIN, FRIEND_REPLY_DELAY_COLD_MAX)

            db = get_db_session()
            try:
                user_meta = db.query(UserMetadata).filter(UserMetadata.user_id == user_id).first()
                if user_meta and user_meta.last_message_date:
                    delta = datetime.utcnow() - user_meta.last_message_date
                    if delta.total_seconds() < FRIEND_REPLY_WARM_WINDOW_MINUTES * 60:
                        delay_seconds = random.randint(
                            FRIEND_REPLY_DELAY_WARM_MIN, FRIEND_REPLY_DELAY_WARM_MAX
                        )
            except Exception as e:
                logger.error(f"DB Error checking last message: {e}")
            finally:
                db.close()
                
            logger.info(f"Waiting {delay_seconds}s before reading message from {user_id}")
            await asyncio.sleep(delay_seconds)
            
            # Mark message as read (after delay above — until here, messages stayed "unread")
            await client.send_read_acknowledge(last_event.chat_id)

            # Pause as if reading the text on screen, before typing starts
            think_s = random.randint(FRIEND_THINKING_AFTER_READ_MIN, FRIEND_THINKING_AFTER_READ_MAX)
            await asyncio.sleep(think_s)

            action = "typing" if not final_media_path else "document"
            async with client.action(last_event.chat_id, action):
                reply_text = await asyncio.to_thread(
                    generate_reply, user_id, combined_text, final_media_path
                )
            
            # Split reply into multiple messages (by newlines or sentence boundaries)
            # Don't split too aggressively. Only split by newlines, or if the text is very long, by punctuation.
            # But avoid splitting every single short sentence.
            parts = [p.strip() for p in re.split(r'\n+', reply_text) if p.strip()]
            
            final_parts = []
            for p in parts:
                # Only split into sentences if the paragraph is longer than 80 chars
                if len(p) > 80 and re.search(r'[.!?]\s', p):
                    # We split by sentence endings but try to group short sentences back together?
                    # For simplicity, let's just not split sentences unless it's a huge block of text.
                    # Or better yet, just send it as is if it's less than ~150 chars.
                    if len(p) < 150:
                        final_parts.append(p)
                    else:
                        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p) if s.strip()]
                        final_parts.extend(sentences)
                else:
                    final_parts.append(p)
            
            # Further parts: short typing before each (first part was "typed" during generation)
            for i, part in enumerate(final_parts):
                if i > 0:
                    typing_delay = min(len(part) / 8, FRIEND_TYPING_PER_PART_MAX)
                    action = "typing" if not final_media_path else "document"
                    async with client.action(last_event.chat_id, action):
                        await asyncio.sleep(typing_delay)

                if i == 0:
                    await last_event.reply(part)
                else:
                    await last_event.respond(part)
                
            # Cleanup downloaded media
            for media_path in media_paths:
                if media_path:
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
                user_meta.consecutive_bot_messages = 0 # Reset because user replied!
                # next_check_date can be set here if we want to reset proactive timer
                db.commit()
            except Exception as e:
                db.rollback()
                logger.error(f"DB Error: {e}")
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error generating reply: {e}")

    except Exception as e:
        logger.error(f"Error in process_buffered_messages: {e}")
    finally:
        setattr(client, f"_processing_{user_id}", False)
        if message_buffers.get(user_id):
            setattr(client, f"_processing_{user_id}", True)
            asyncio.create_task(process_buffered_messages(client, user_id))
