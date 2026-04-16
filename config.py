import os
from dotenv import load_dotenv

load_dotenv()

# Telegram API
API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "")

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Qdrant Database
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Other configuration
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "3600")) # How often to run proactive loop
MONITORED_CHANNELS = [c.strip() for c in os.getenv("MONITORED_CHANNELS", "").split(",") if c.strip()]
FRIENDS_LIST = [f.strip() for f in os.getenv("FRIENDS_LIST", "").split(",") if f.strip()]
