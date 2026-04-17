import importlib
import os


def test_config_defaults():
    for key in (
        "API_ID",
        "API_HASH",
        "GEMINI_API_KEY",
        "MONITORED_CHANNELS",
        "FRIENDS_LIST",
        "RECENT_CHAT_WINDOW_MESSAGES",
    ):
        os.environ.pop(key, None)
    import config

    importlib.reload(config)
    assert config.API_ID == 0
    assert config.API_HASH == ""
    assert config.GEMINI_API_KEY == ""
    assert config.MONITORED_CHANNELS == []
    assert config.FRIENDS_LIST == []
    assert config.RECENT_CHAT_WINDOW_MESSAGES == 10


def test_config_parses_lists_and_ints(monkeypatch):
    monkeypatch.setenv("API_ID", "12345")
    monkeypatch.setenv("MONITORED_CHANNELS", " foo , bar ")
    monkeypatch.setenv("FRIENDS_LIST", "u1,u2")
    monkeypatch.setenv("QDRANT_PORT", "9000")
    monkeypatch.setenv("BUFFER_QUIET_SECONDS", "7.5")
    import config

    importlib.reload(config)
    assert config.API_ID == 12345
    assert config.MONITORED_CHANNELS == ["foo", "bar"]
    assert config.FRIENDS_LIST == ["u1", "u2"]
    assert config.QDRANT_PORT == 9000
    assert config.BUFFER_QUIET_SECONDS == 7.5


def test_config_reply_delay_bounds(monkeypatch):
    monkeypatch.setenv("FRIEND_REPLY_DELAY_COLD_MIN", "10")
    monkeypatch.setenv("FRIEND_REPLY_DELAY_COLD_MAX", "20")
    monkeypatch.setenv("FRIEND_REPLY_WARM_WINDOW_MINUTES", "30")
    import config

    importlib.reload(config)
    assert config.FRIEND_REPLY_DELAY_COLD_MIN == 10
    assert config.FRIEND_REPLY_DELAY_COLD_MAX == 20
    assert config.FRIEND_REPLY_WARM_WINDOW_MINUTES == 30
