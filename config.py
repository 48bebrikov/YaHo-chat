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
# Do not send proactive messages if the user wrote in private within this many minutes (avoids interrupting an active chat)
PROACTIVE_MIN_IDLE_MINUTES = int(os.getenv("PROACTIVE_MIN_IDLE_MINUTES", "90"))
MONITORED_CHANNELS = [c.strip() for c in os.getenv("MONITORED_CHANNELS", "").split(",") if c.strip()]
FRIENDS_LIST = [f.strip() for f in os.getenv("FRIENDS_LIST", "").split(",") if f.strip()]

# Private chat: merge several user messages into one reply (seconds of silence before processing)
BUFFER_QUIET_SECONDS = float(os.getenv("BUFFER_QUIET_SECONDS", "5"))
# Simulated delay before "reading" and replying (seconds)
FRIEND_REPLY_DELAY_COLD_MIN = int(os.getenv("FRIEND_REPLY_DELAY_COLD_MIN", "60"))
FRIEND_REPLY_DELAY_COLD_MAX = int(os.getenv("FRIEND_REPLY_DELAY_COLD_MAX", "600"))
FRIEND_REPLY_DELAY_WARM_MIN = int(os.getenv("FRIEND_REPLY_DELAY_WARM_MIN", "5"))
FRIEND_REPLY_DELAY_WARM_MAX = int(os.getenv("FRIEND_REPLY_DELAY_WARM_MAX", "30"))
# If last bot–user message was within this many minutes, use WARM delay range
FRIEND_REPLY_WARM_WINDOW_MINUTES = int(os.getenv("FRIEND_REPLY_WARM_WINDOW_MINUTES", "15"))
# After messages are marked read: pause before typing/generation (simulates reading on screen)
FRIEND_THINKING_AFTER_READ_MIN = int(os.getenv("FRIEND_THINKING_AFTER_READ_MIN", "3"))
FRIEND_THINKING_AFTER_READ_MAX = int(os.getenv("FRIEND_THINKING_AFTER_READ_MAX", "12"))
# Extra "typing" time per outgoing message part after the first (seconds cap)
FRIEND_TYPING_PER_PART_MAX = float(os.getenv("FRIEND_TYPING_PER_PART_MAX", "10"))
