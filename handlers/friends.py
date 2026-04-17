import asyncio
import logging
import os
import random
import re
from datetime import datetime, timezone

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
from database.sqlite_db import db_session, UserMetadata
logger = logging.getLogger(__name__)

from collections import defaultdict
user_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

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
    try:
        with db_session() as db:
            user_meta = db.query(UserMetadata).filter(UserMetadata.user_id == user_id).first()
            if not user_meta:
                user_meta = UserMetadata(user_id=user_id)
                db.add(user_meta)
            user_meta.last_user_message_at = datetime.now(timezone.utc)
    except Exception as e:
        logger.error(f"DB Error recording user activity: {e}")

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
        if user_locks[user_id].locked():
            return
            
        asyncio.create_task(_run_processing(client, user_id))

async def _run_processing(client, user_id: str):
    async with user_locks[user_id]:
        while message_buffers.get(user_id):
            await process_buffered_messages(client, user_id)

async def process_buffered_messages(client, user_id: str):
    media_paths = []
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

            try:
                with db_session() as db:
                    user_meta = db.query(UserMetadata).filter(UserMetadata.user_id == user_id).first()
                    
                    # Make last_message_date offset-aware if it's naive
                    if user_meta and user_meta.last_message_date:
                        last_msg_date = user_meta.last_message_date
                        if last_msg_date.tzinfo is None:
                            last_msg_date = last_msg_date.replace(tzinfo=timezone.utc)
                            
                        delta = datetime.now(timezone.utc) - last_msg_date
                        if delta.total_seconds() < FRIEND_REPLY_WARM_WINDOW_MINUTES * 60:
                            delay_seconds = random.randint(
                                FRIEND_REPLY_DELAY_WARM_MIN, FRIEND_REPLY_DELAY_WARM_MAX
                            )
            except Exception as e:
                logger.error(f"DB Error checking last message: {e}")
                
            logger.info(f"Waiting {delay_seconds}s before reading message from {user_id}")
            await asyncio.sleep(delay_seconds)
            
            # Mark message as read (after delay above — until here, messages stayed "unread")
            await client.send_read_acknowledge(last_event.chat_id)

            # Pause as if reading the text on screen, before typing starts
            think_s = random.randint(FRIEND_THINKING_AFTER_READ_MIN, FRIEND_THINKING_AFTER_READ_MAX)
            await asyncio.sleep(think_s)

            action = "typing" if not final_media_path else "document"
            async with client.action(last_event.chat_id, action):
                try:
                    reply_text = await asyncio.wait_for(
                        generate_reply(user_id, combined_text, final_media_path),
                        timeout=90.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"generate_reply timed out for {user_id}")
                    return
            
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
            
            from ai.metrics import BOT_MESSAGES_SENT_TOTAL
            BOT_MESSAGES_SENT_TOTAL.labels(type="reply").inc()
                
            # Update user metadata in SQLite
            try:
                with db_session() as db:
                    user_meta = db.query(UserMetadata).filter(UserMetadata.user_id == user_id).first()
                    if not user_meta:
                        user_meta = UserMetadata(user_id=user_id)
                        db.add(user_meta)
                    
                    user_meta.last_message_date = datetime.now(timezone.utc)
                    user_meta.consecutive_bot_messages = 0 # Reset because user replied!
                    # next_check_date can be set here if we want to reset proactive timer
            except Exception as e:
                logger.error(f"DB Error: {e}")

        except Exception as e:
            logger.error(f"Error generating reply: {e}")

    except Exception as e:
        logger.error(f"Error in process_buffered_messages: {e}")
    finally:
        for media_path in media_paths:
            if media_path:
                try:
                    os.remove(media_path)
                except Exception as e:
                    logger.error(f"Failed to delete {media_path}: {e}")
