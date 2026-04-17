"""Второй вызов Gemini: что положить в RAG (facts vs dialogue_snippet)."""

import json
import logging
import re

from google.genai import types

logger = logging.getLogger(__name__)

MEMORY_EXTRACTOR_SYSTEM = """Ты анализируешь одну пару сообщений в личном чате: реплика друга и твой ответ.
Реши, что стоит сохранить в долговременную память бота для этого друга.

Типы записей:
- kind "fact" — устойчивые сведения о друге, его жизни, предпочтениях, планах (город, работа, что любит/не любит). Короткие формулировки, без воды.
- kind "dialogue_snippet" — важная формулировка из переписки, которую нельзя сжать до факта без потери смысла (договорённость, эмоционально важная фраза, цитата). Можно взять суть из сообщения друга или из твоего ответа.

Правила:
- Если сообщение пустое, только смайлики/«ок»/«да» без содержания — можно ничего не сохранять (items: []).
- Не дублируй одно и то же разными формулировками.
- Не придумывай фактов, которых нет в тексте.
- Ответ строго JSON без markdown-обёртки."""

MEMORY_EXTRACTOR_USER_TEMPLATE = """Текущее время события (UTC): {event_utc_iso}

Сообщение друга:
{user_message}

Твой ответ:
{bot_reply}

Верни JSON вида:
{{
  "skip_all": false,
  "items": [
    {{"kind": "fact", "text": "...", "source": "user"}},
    {{"kind": "dialogue_snippet", "text": "...", "source": "bot"}}
  ]
}}

source — откуда взята суть: "user" или "bot".
Если ничего сохранять не нужно: {{"skip_all": true, "items": []}}"""


def _parse_json_from_model(text: str) -> dict:
    raw = (text or "").strip()
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL | re.I)
        if m:
            raw = m.group(1).strip()
    return json.loads(raw)


def extract_memory_items(user_message: str, bot_reply: str, event_utc_iso: str) -> dict | None:
    """
    Возвращает dict с ключами skip_all, items.
    При ошибке парсинга/API — None (вызывающий делает fallback).
    """
    from ai.gemini_engine import MODEL_ID, generate_content_with_retry_sync, get_genai_client

    client = get_genai_client()
    user_part = MEMORY_EXTRACTOR_USER_TEMPLATE.format(
        event_utc_iso=event_utc_iso,
        user_message=user_message or "",
        bot_reply=bot_reply or "",
    )
    response = generate_content_with_retry_sync(
        client,
        model=MODEL_ID,
        contents=types.UserContent(user_part),
        config=types.GenerateContentConfig(
            system_instruction=MEMORY_EXTRACTOR_SYSTEM,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    raw = response.text or ""
    try:
        data = _parse_json_from_model(raw)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Memory extraction JSON parse failed: %s — raw: %s", e, raw[:500])
        return None
    if not isinstance(data, dict):
        return None
    return data


def normalize_items(data: dict) -> list[dict]:
    """Валидирует и ограничивает items."""
    if data.get("skip_all"):
        return []
    items = data.get("items") or []
    if not isinstance(items, list):
        return []
    out = []
    for it in items[:12]:
        if not isinstance(it, dict):
            continue
        kind = it.get("kind")
        text = (it.get("text") or "").strip()
        source = it.get("source") or "user"
        if kind not in ("fact", "dialogue_snippet"):
            continue
        if source not in ("user", "bot"):
            source = "user"
        if not text or len(text) > 800:
            continue
        out.append({"kind": kind, "text": text, "source": source})
    return out
