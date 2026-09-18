from __future__ import annotations

import logging
import os

from i18n.en import COPY as COPY_EN
from i18n.ru import COPY as COPY_RU
from i18n.schema import BotCopy

logger = logging.getLogger(__name__)

PACKS: dict[str, BotCopy] = {
    "ru": COPY_RU,
    "en": COPY_EN,
}


def resolve_locale(locale: str | None = None) -> str:
    raw = locale if locale is not None else os.getenv("BOT_LOCALE", "ru")
    key = (raw or "ru").strip().lower()
    if key not in PACKS:
        logger.warning("Unknown BOT_LOCALE=%r, falling back to ru", raw)
        return "ru"
    return key


def get_copy(locale: str | None = None) -> BotCopy:
    """Return the copy pack for BOT_LOCALE (or an explicit locale)."""
    return PACKS[resolve_locale(locale)]
