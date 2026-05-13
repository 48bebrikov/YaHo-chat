import os
from dotenv import load_dotenv

load_dotenv()

# Telegram API
API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "")

# OpenRouter API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL_ID = os.getenv("OPENROUTER_MODEL_ID", "moonshotai/kimi-k2.6")

# Gemini API (Still used for TTS / Voice messages)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-3.1-flash-lite")
TTS_VOICE = os.getenv("TTS_VOICE", "Achernar") # "Aoede", "Callirrhoe", "Kore", "Charon", etc.

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Qdrant Database
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Other configuration
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "3600")) # How often to run proactive loop
# Do not send proactive messages if the user wrote in private within this many minutes (avoids interrupting an active chat)
PROACTIVE_MIN_IDLE_MINUTES = int(os.getenv("PROACTIVE_MIN_IDLE_MINUTES", "90"))
# IANA timezone for proactive prompts and night-time skip (default matches previous fixed GMT+7).
PROACTIVE_LOCAL_TIMEZONE = os.getenv("PROACTIVE_LOCAL_TIMEZONE", "Asia/Bangkok")
# After a proactive message is sent, wait at least this many hours before the next check for this friend.
PROACTIVE_COOLDOWN_HOURS_AFTER_SEND = int(os.getenv("PROACTIVE_COOLDOWN_HOURS_AFTER_SEND", "6"))
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

# Verbatim recent chat lines (user+bot) passed to Gemini alongside RAG
RECENT_CHAT_WINDOW_MESSAGES = int(os.getenv("RECENT_CHAT_WINDOW_MESSAGES", "10"))
