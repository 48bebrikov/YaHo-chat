from dataclasses import fields

from i18n import get_copy, resolve_locale
from i18n.en import COPY as COPY_EN
from i18n.ru import COPY as COPY_RU
from i18n.schema import BotCopy


def test_default_locale_is_ru(monkeypatch):
    monkeypatch.delenv("BOT_LOCALE", raising=False)
    copy = get_copy()
    assert copy.locale == "ru"
    assert copy.llm_call_fallback == "Прости, что-то с интернетом, не могу ответить."
    assert copy.agent_run_fallback == "Блин, чет интернет отвалился..."


def test_en_locale(monkeypatch):
    monkeypatch.setenv("BOT_LOCALE", "en")
    copy = get_copy()
    assert copy.locale == "en"
    assert "internet" in copy.llm_call_fallback.lower()
    assert copy.memory_extractor_system.startswith("You are analyzing")


def test_unknown_locale_falls_back_to_ru(monkeypatch):
    monkeypatch.setenv("BOT_LOCALE", "de")
    assert resolve_locale() == "ru"
    assert get_copy().locale == "ru"


def test_explicit_locale_overrides_env(monkeypatch):
    monkeypatch.setenv("BOT_LOCALE", "ru")
    assert get_copy("en").locale == "en"


def test_packs_share_schema():
    names = {f.name for f in fields(BotCopy)}
    assert set(COPY_RU.__dataclass_fields__) == names
    assert set(COPY_EN.__dataclass_fields__) == names


def test_reminder_template_formats():
    msg = COPY_EN.reminder_template.format(text="call mom")
    assert "call mom" in msg
    msg_ru = COPY_RU.reminder_template.format(text="позвонить маме")
    assert "позвонить маме" in msg_ru
