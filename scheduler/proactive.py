import asyncio
import logging
import json
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from database.sqlite_db import db_session, UserMetadata, NewsCache
from config import (
    POLL_INTERVAL_SECONDS,
    FRIENDS_LIST,
    PROACTIVE_MIN_IDLE_MINUTES,
    PROACTIVE_LOCAL_TIMEZONE,
    PROACTIVE_COOLDOWN_HOURS_AFTER_SEND,
)
from google.genai import types

from ai.gemini_engine import (
    MODEL_ID,
    complete_tool_response,
    config_with_tools,
    generate_content_with_retry,
    get_genai_client,
)
from ai.rag import get_memory_context
logger = logging.getLogger(__name__)

PROACTIVE_PROMPT_GENERAL = """You are deciding whether to proactively text a friend on Telegram.
You are an real human friend.
There are NO new channel posts queued for them right now (you cannot forward news in this turn).
Context:
- User ID: {user_id}
- Hours since last interaction: {hours_since}
- Consecutive messages you sent without reply: {consecutive_messages}
- Friend local date & time: {friend_local_time}
- Reference UTC: {current_time_utc}

Consider:
1. Don't spam. If it's been less than a few hours, probably don't text unless something important. (The system already avoids pinging while the user was recently active in DM.)
2. If it's been a day or more, maybe just say hi.
3. If consecutive_messages == 1 and hours_since > 12, you can send ONE follow up like "ауу", "ты тут?", "игноришь?".
4. If consecutive_messages >= 2, STOP MESSAGING THEM completely (return should_message=false). Don't be annoying.
5. Be natural and write in Russian.
6. Use friend local time for morning/evening/night small talk — not UTC alone. Do not say it is "night" if local time is morning or daytime.
7. Do not repeat the same opening hook or catchphrase as in a recent proactive line; vary wording.

You must respond in valid JSON format ONLY, with this structure:
{{
    "should_message": true or false,
    "message_text": "text to send if true, or empty string if false",
    "next_check_hours": integer (how many hours to wait before checking again if false, usually 1 to 24)
}}
"""

PROACTIVE_PROMPT_WITH_FORWARD = """You help decide whether to ping a friend and what SHORT personal line to add in Russian.
IMPORTANT: The original post from the Telegram channel will be FORWARDED to them as-is (same channel, link, media). You must NOT repeat, summarize, or retell the news — they will read the real post.
Your job is ONLY an optional 1–2 sentence casual follow-up (or empty if forwarding alone is enough), like a real friend reacting to what you shared.
Context:
- User ID: {user_id}
- Hours since last interaction: {hours_since}
- Consecutive messages you sent without reply: {consecutive_messages}
- News post preview (tone only, do not copy): {news_preview}
- Friend local date & time: {friend_local_time}
- Reference UTC: {current_time_utc}

Rules:
1. Same anti-spam rules as usual: if consecutive_messages >= 2, return should_message=false.
2. message_text must be ONLY your short reaction/comment, NOT the article text.
3. If a forward alone is enough, set message_text to "".
4. Use friend local time for tone (morning coffee vs late evening). Do not treat UTC as their local "night" or "morning".
5. Do not reuse the same opening hook as a recent proactive message; vary phrasing.

Respond in valid JSON ONLY:
{{
    "should_message": true or false,
    "message_text": "short Russian comment or empty string",
    "next_check_hours": integer
}}
"""


def _proactive_tz() -> ZoneInfo:
    try:
        return ZoneInfo(PROACTIVE_LOCAL_TIMEZONE)
    except Exception:
        logger.warning(
            "Invalid PROACTIVE_LOCAL_TIMEZONE %r; using Asia/Bangkok",
            PROACTIVE_LOCAL_TIMEZONE,
        )
        return ZoneInfo("Asia/Bangkok")


def _pick_news_for_friend(db, user_meta) -> NewsCache | None:
    """Next unseen news for this friend."""
    if not user_meta:
        return None
    if not user_meta.last_forwarded_news_id:
        from sqlalchemy import func
        max_id = db.query(func.max(NewsCache.id)).scalar() or 0
        user_meta.last_forwarded_news_id = max_id
        return None

    last_id = user_meta.last_forwarded_news_id
    pending = (
        db.query(NewsCache)
        .filter(NewsCache.id > last_id)
        .order_by(NewsCache.id.asc())
        .limit(1)
        .all()
    )
    if not pending:
        return None
    return pending[0]


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

    try:
        gemini_client = get_genai_client()
    except RuntimeError:
        logger.error("GEMINI_API_KEY missing; proactive check skipped.")
        return

    try:
        with db_session() as db:
            current_utc = datetime.now(timezone.utc)
            proactive_tz = _proactive_tz()
            now_local = current_utc.astimezone(proactive_tz)
            if 3 <= now_local.hour < 8:
                logger.info(
                    "It's night time in %s (%s). Skipping proactive check.",
                    proactive_tz.key,
                    now_local.strftime("%H:%M"),
                )
                return

            tz_label = proactive_tz.key
            friend_local_time = (
                f"{now_local.strftime('%Y-%m-%d %H:%M')} ({tz_label}, friend local time)"
            )
            current_time_utc = current_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

            for friend_index, friend_id in enumerate(FRIENDS_LIST):
                user_meta = db.query(UserMetadata).filter(UserMetadata.user_id == friend_id).first()

                if user_meta:
                    if user_meta.next_check_date:
                        next_check = user_meta.next_check_date
                        if next_check.tzinfo is None:
                            next_check = next_check.replace(tzinfo=timezone.utc)
                        if datetime.now(timezone.utc) < next_check:
                            continue

                    if user_meta.last_user_message_at:
                        last_user_msg = user_meta.last_user_message_at
                        if last_user_msg.tzinfo is None:
                            last_user_msg = last_user_msg.replace(tzinfo=timezone.utc)
                            
                        idle_minutes = (
                            datetime.now(timezone.utc) - last_user_msg
                        ).total_seconds() / 60
                        if idle_minutes < PROACTIVE_MIN_IDLE_MINUTES:
                            logger.info(
                                f"Skipping proactive for {friend_id}: user wrote {idle_minutes:.0f} min ago "
                                f"(threshold {PROACTIVE_MIN_IDLE_MINUTES} min)"
                            )
                            continue

                hours_since = 24
                consecutive_messages = 0
                if user_meta:
                    last_msg_date = user_meta.last_message_date
                    if last_msg_date and last_msg_date.tzinfo is None:
                        last_msg_date = last_msg_date.replace(tzinfo=timezone.utc)
                        
                    if last_msg_date:
                        delta = datetime.now(timezone.utc) - last_msg_date
                        hours_since = round(delta.total_seconds() / 3600, 1)
                        if hours_since > 24 * 7:
                            user_meta.consecutive_bot_messages = 0
                    consecutive_messages = user_meta.consecutive_bot_messages or 0

                next_news = _pick_news_for_friend(db, user_meta)

                context = get_memory_context(
                    friend_id, "latest conversation", limit_facts=3, limit_dialogue=3
                )

                if next_news:
                    preview = (next_news.text or "")[:800]
                    prompt = PROACTIVE_PROMPT_WITH_FORWARD.format(
                        user_id=friend_id,
                        hours_since=hours_since,
                        consecutive_messages=consecutive_messages,
                        news_preview=preview,
                        friend_local_time=friend_local_time,
                        current_time_utc=current_time_utc,
                    )
                else:
                    prompt = PROACTIVE_PROMPT_GENERAL.format(
                        user_id=friend_id,
                        hours_since=hours_since,
                        consecutive_messages=consecutive_messages,
                        friend_local_time=friend_local_time,
                        current_time_utc=current_time_utc,
                    )

                if context:
                    prompt += f"\n\nRecent conversation context:\n{context}"

                prompt_content = [prompt]
                if next_news and next_news.media_path:
                    import os
                    if os.path.exists(next_news.media_path):
                        import PIL.Image
                        try:
                            img = PIL.Image.open(next_news.media_path)
                            prompt_content.append(img)
                        except Exception as e:
                            logger.error(f"Failed to open image {next_news.media_path}: {e}")

                first_user = types.UserContent(prompt_content)
                response = await generate_content_with_retry(
                    gemini_client,
                    model=MODEL_ID,
                    contents=first_user,
                    config=config_with_tools(),
                )

                if response.function_calls:
                    logger.info(
                        f"Proactive Gemini decided to use tool: {response.function_calls[0].name}"
                    )
                    response = await complete_tool_response(gemini_client, first_user, response)

                try:
                    text_response = response.text
                    json_match = re.search(r"```json\n(.*?)\n```", text_response, re.DOTALL)
                    if json_match:
                        text_response = json_match.group(1)

                    decision = json.loads(text_response)
                    should_message = decision.get("should_message", False)
                    message_text = (decision.get("message_text") or "").strip()
                    next_check_hours = decision.get("next_check_hours", 4)

                    if not user_meta:
                        user_meta = UserMetadata(user_id=friend_id)
                        db.add(user_meta)

                    try:
                        target = int(friend_id)
                    except ValueError:
                        target = friend_id

                    delivered = False
                    if should_message and (next_news or message_text):
                        if next_news:
                            try:
                                from_peer = await client.get_entity(next_news.channel_id)
                                await client.forward_messages(target, next_news.message_id, from_peer=from_peer)
                                logger.info(
                                    f"Forwarded news id={next_news.id} from {next_news.channel_id} to {target}"
                                )
                                delivered = True
                            except Exception as ex:
                                logger.error(
                                    f"Forward failed for news id={next_news.id}: {ex}"
                                )

                    if message_text:
                        logger.info(f"Proactively sending comment to {target}: {message_text[:120]}...")
                        await client.send_message(target, message_text)
                        delivered = True

                    if delivered:
                        from ai.metrics import BOT_MESSAGES_SENT_TOTAL
                        BOT_MESSAGES_SENT_TOTAL.labels(type="proactive").inc()

                        if next_news:
                            user_meta.last_forwarded_news_id = next_news.id

                        user_meta.last_message_date = datetime.now(timezone.utc)
                        user_meta.consecutive_bot_messages = (user_meta.consecutive_bot_messages or 0) + 1
                        cooldown_h = max(1, PROACTIVE_COOLDOWN_HOURS_AFTER_SEND)
                        user_meta.next_check_date = datetime.now(timezone.utc) + timedelta(
                            hours=cooldown_h
                        )
                    else:
                        logger.info(
                            f"Decided not to message {friend_id}. Checking again in {next_check_hours} hours."
                        )
                        user_meta.next_check_date = datetime.now(timezone.utc) + timedelta(hours=next_check_hours)

                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from Gemini: {response.text}")

    except Exception as e:
        logger.error(f"Failed during check_and_message_friends: {e}")
