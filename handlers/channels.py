from telethon import events
from database.sqlite_db import db_session, NewsCache
import logging

logger = logging.getLogger(__name__)

def register_channel_handlers(client, monitored_channels: list[str]):
    if not monitored_channels or monitored_channels == [""]:
        logger.info("No channels to monitor.")
        return

    @client.on(events.NewMessage(chats=monitored_channels))
    async def handler(event):
        """Saves new posts from monitored channels."""
        if not event.text:
            return # ignore media-only for now

        channel = await event.get_chat()
        channel_id = getattr(channel, 'username', str(channel.id))
        
        try:
            with db_session() as db:
                news = NewsCache(
                    channel_id=channel_id,
                    message_id=event.id,
                    text=event.text
                )
                db.add(news)
            logger.info(f"Saved news from {channel_id}: {event.text[:30]}...")
        except Exception as e:
            logger.error(f"Failed to save news: {e}")
