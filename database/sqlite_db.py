import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

# Local SQLite Database setup
DB_URL = "sqlite:///local_cache.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class NewsCache(Base):
    """Stores posts from monitored channels to be used as context or forwarded."""
    __tablename__ = "news_cache"

    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    message_id = Column(Integer)
    text = Column(Text)
    date_added = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class FriendChatLog(Base):
    """Дословный лог личных сообщений user/bot для последних N реплик в промпте (не RAG)."""

    __tablename__ = "friend_chat_log"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    role = Column(String)  # "user" | "bot"
    text = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class UserMetadata(Base):
    """Stores metadata about friends we are talking to, to decide when to message them."""
    __tablename__ = "users_metadata"

    user_id = Column(String, primary_key=True, index=True)
    last_message_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_topic = Column(Text, nullable=True)
    next_check_date = Column(DateTime, nullable=True) # Set by AI when to check next
    consecutive_bot_messages = Column(Integer, default=0) # Tracks if the user is ignoring us
    # Last news_cache row we forwarded to this friend (proactive); next pick is id > this
    last_forwarded_news_id = Column(Integer, nullable=True)
    # Updated immediately when the user sends a private message (before bot reply). Used to avoid proactive pings mid-conversation.
    last_user_message_at = Column(DateTime, nullable=True)

# Create all tables
Base.metadata.create_all(bind=engine)

# Quick migration if table exists without the new column
try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE users_metadata ADD COLUMN consecutive_bot_messages INTEGER DEFAULT 0"))
        conn.commit()
except Exception:
    pass # Column already exists or other error

try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE users_metadata ADD COLUMN last_forwarded_news_id INTEGER"))
        conn.commit()
except Exception:
    pass

try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE users_metadata ADD COLUMN last_user_message_at DATETIME"))
        conn.commit()
except Exception:
    pass

from contextlib import contextmanager

@contextmanager
def db_session():
    """Context manager for DB sessions."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def append_friend_chat_turn(user_id: str, user_text: str, bot_text: str) -> None:
    """Добавляет пару реплик (друг → бот) после ответа."""
    db = SessionLocal()
    try:
        db.add(FriendChatLog(user_id=user_id, role="user", text=user_text or ""))
        db.add(FriendChatLog(user_id=user_id, role="bot", text=bot_text or ""))
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def fetch_previous_friend_messages(user_id: str, limit: int) -> list[tuple[str, str]]:
    """Последние `limit` реплик по id (каждая строка — user или bot), по времени снизу вверх → возвращаем в хронологическом порядке."""
    if limit <= 0:
        return []
    db = SessionLocal()
    try:
        rows = (
            db.query(FriendChatLog)
            .filter(FriendChatLog.user_id == user_id)
            .order_by(FriendChatLog.id.desc())
            .limit(limit)
            .all()
        )
        rows = list(reversed(rows))
        return [(r.role, r.text or "") for r in rows]
    finally:
        db.close()
