from database.sqlite_db import (
    FriendChatLog,
    NewsCache,
    SessionLocal,
    append_friend_chat_turn,
    fetch_previous_friend_messages,
)


def test_newscache_roundtrip():
    db = SessionLocal()
    try:
        row = NewsCache(channel_id="test_channel_ci", message_id=999001, text="pytest body")
        db.add(row)
        db.commit()
        rid = row.id
        loaded = db.query(NewsCache).filter(NewsCache.id == rid).first()
        assert loaded is not None
        assert loaded.text == "pytest body"
        db.delete(loaded)
        db.commit()
    finally:
        db.close()


def test_friend_chat_log_append_and_fetch():
    uid = "test_user_chat_ci"
    db = SessionLocal()
    try:
        db.query(FriendChatLog).filter(FriendChatLog.user_id == uid).delete()
        db.commit()
    finally:
        db.close()

    append_friend_chat_turn(uid, "привет", "здрасьте")
    prev = fetch_previous_friend_messages(uid, 9)
    assert len(prev) == 2
    assert prev[0] == ("user", "привет")
    assert prev[1] == ("bot", "здрасьте")

    db = SessionLocal()
    try:
        db.query(FriendChatLog).filter(FriendChatLog.user_id == uid).delete()
        db.commit()
    finally:
        db.close()


def test_get_db_session_returns_usable_session():
    from database.sqlite_db import db_session, NewsCache

    with db_session() as db:
        n = db.query(NewsCache).count()
        assert isinstance(n, int)
