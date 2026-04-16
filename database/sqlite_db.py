import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

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
    date_added = Column(DateTime, default=datetime.utcnow)

class UserMetadata(Base):
    """Stores metadata about friends we are talking to, to decide when to message them."""
    __tablename__ = "users_metadata"

    user_id = Column(String, primary_key=True, index=True)
    last_message_date = Column(DateTime, default=datetime.utcnow)
    last_topic = Column(Text, nullable=True)
    next_check_date = Column(DateTime, nullable=True) # Set by AI when to check next
    consecutive_bot_messages = Column(Integer, default=0) # Tracks if the user is ignoring us

# Create all tables
Base.metadata.create_all(bind=engine)

# Quick migration if table exists without the new column
try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE users_metadata ADD COLUMN consecutive_bot_messages INTEGER DEFAULT 0"))
        conn.commit()
except Exception:
    pass # Column already exists or other error

def get_db_session():
    """Returns a new DB session."""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()
