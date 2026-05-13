import asyncio
import logging
from telethon import TelegramClient
from config import API_ID, API_HASH, MONITORED_CHANNELS, FRIENDS_LIST

from handlers.channels import register_channel_handlers
from handlers.friends import register_friend_handlers
from scheduler.proactive import proactive_loop, reminder_loop

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting Telegram AI Friend Userbot...")
    
    if not API_ID or not API_HASH:
        logger.error("Please set API_ID and API_HASH in your .env file.")
        return

    client = TelegramClient('userbot_session', API_ID, API_HASH)
    
    # Register handlers
    register_channel_handlers(client, MONITORED_CHANNELS)
    register_friend_handlers(client, FRIENDS_LIST)
    
    # Start Prometheus metrics server
    from prometheus_client import start_http_server
    try:
        start_http_server(8000)
        logger.info("Prometheus metrics server started on port 8000")
    except Exception as e:
        logger.error(f"Failed to start Prometheus metrics server: {e}")

    await client.start()
    logger.info("Userbot started successfully.")
    
    # Start the proactive background loop
    client.loop.create_task(proactive_loop(client))
    client.loop.create_task(reminder_loop(client))
    
    # Run the client until disconnected
    await client.run_until_disconnected()

if __name__ == "__main__":
    asyncio.run(main())
