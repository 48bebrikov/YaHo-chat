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
        if not event.text and not event.photo:
            return # ignore non-text and non-photo media for now

        channel = await event.get_chat()
        channel_id = getattr(channel, 'username', str(channel.id))
        
        media_path = None
        if event.photo:
            try:
                media_path = await event.download_media(file="database/")
                logger.info(f"Downloaded media to {media_path}")
            except Exception as e:
                logger.error(f"Failed to download media: {e}")
        
        try:
            with db_session() as db:
                news = NewsCache(
                    channel_id=channel_id,
                    message_id=event.id,
                    text=event.text or "",
                    media_path=media_path
                )
                db.add(news)
            text_preview = (event.text or "")[:30]
            logger.info(f"Saved news from {channel_id}: {text_preview}...")
        except Exception as e:
            logger.error(f"Failed to save news: {e}")
