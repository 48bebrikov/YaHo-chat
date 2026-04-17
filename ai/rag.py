import logging
import time

from ai.embedder import embedder
from database.qdrant_db import qdrant_db

logger = logging.getLogger(__name__)

LIMIT_FACTS = 8
LIMIT_DIALOGUE = 6


def format_recent_chat_block(user_id: str, current_message: str, max_lines: int) -> str:
    """Дословные последние реплики из SQLite + текущее сообщение друга (до max_lines строк)."""
    from database.sqlite_db import fetch_previous_friend_messages

    n_prev = max(0, max_lines - 1)
    prev = fetch_previous_friend_messages(user_id, n_prev)
    lines = []
    for role, text in prev:
        label = "Friend" if role == "user" else "You"
        lines.append(f"{label}: {text}")
    lines.append(f"Friend: {current_message}")
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return (
        "Recent conversation (verbatim, chronological; last line is the friend's latest message):\n"
        + "\n".join(lines)
    )


def _sort_key(payload: dict) -> float:
    return float(payload.get("sort_ts") or payload.get("timestamp") or 0.0)


def _format_label(event_utc_iso: str | None, role: str, text: str, kind: str) -> str:
    prefix = "FACT" if kind == "fact" else ("DIALOGUE" if kind == "dialogue_snippet" else "MEM")
    role_label = "Friend" if role == "user" else "You"
    ts = (event_utc_iso or "").strip() or "?"
    return f"[{prefix}] [{ts}] {role_label}: {text}"


def get_memory_context(
    user_id: str,
    query: str,
    *,
    limit_facts: int = LIMIT_FACTS,
    limit_dialogue: int = LIMIT_DIALOGUE,
) -> str:
    """
    Два независимых поиска по kind: fact и dialogue_snippet, затем объединение по времени.
    """
    query_embedding = embedder.get_embedding(query)
    facts = qdrant_db.search_similar_by_kind(
        user_id, query_embedding, "fact", limit_facts
    )
    dialogue = qdrant_db.search_similar_by_kind(
        user_id, query_embedding, "dialogue_snippet", limit_dialogue
    )
    merged = list(facts) + list(dialogue)
    if not merged:
        return ""

    import time
    cutoff_ts = time.time() - 4 * 3600

    merged = [r for r in merged if _sort_key(r.payload or {}) < cutoff_ts]
    merged.sort(key=lambda r: _sort_key(r.payload or {}))

    lines = []
    for r in merged:
        p = r.payload or {}
        role = p.get("role", "unknown")
        text = p.get("text", "")
        kind = p.get("kind", "")
        event_utc = p.get("event_utc_iso")
        lines.append(_format_label(event_utc, role, text, kind))

    return "\n".join(lines)


def _save_embedding_point(
    user_id: str,
    text: str,
    role: str,
    kind: str,
    sort_ts: float,
    event_utc_iso: str,
) -> None:
    emb = embedder.get_embedding(text)
    qdrant_db.upsert_memory_point(
        user_id=user_id,
        text=text,
        embedding=emb,
        role=role,
        kind=kind,
        sort_ts=sort_ts,
        event_utc_iso=event_utc_iso,
    )


def persist_conversation_turn(
    user_id: str,
    user_message: str,
    bot_reply: str,
    sort_ts: float | None = None,
    event_utc_iso: str | None = None,
) -> None:
    """
    Второй вызов Gemini решает, что сохранить; при ответе None — fallback: два dialogue_snippet.
    """
    if sort_ts is None:
        sort_ts = time.time()
    if event_utc_iso is None:
        if sort_ts:
            from datetime import datetime, timezone

            event_utc_iso = datetime.fromtimestamp(sort_ts, tz=timezone.utc).isoformat()
        else:
            event_utc_iso = ""

    from ai.memory_extraction import extract_memory_items, normalize_items

    try:
        raw = extract_memory_items(user_message, bot_reply, event_utc_iso)
    except Exception:
        logger.exception("Memory extraction failed; using fallback dialogue save")
        _fallback_dialogue_pair(user_id, user_message, bot_reply, sort_ts, event_utc_iso)
        return

    if raw is None:
        _fallback_dialogue_pair(user_id, user_message, bot_reply, sort_ts, event_utc_iso)
        return

    items = normalize_items(raw)
    if raw.get("skip_all") or not items:
        return

    for it in items:
        _save_embedding_point(
            user_id=user_id,
            text=it["text"],
            role=it["source"],
            kind=it["kind"],
            sort_ts=sort_ts,
            event_utc_iso=event_utc_iso,
        )


def _fallback_dialogue_pair(
    user_id: str,
    user_message: str,
    bot_reply: str,
    sort_ts: float,
    event_utc_iso: str,
) -> None:
    """Сохраняет полные реплики как dialogue_snippet (как раньше)."""
    um = (user_message or "").strip()
    br = (bot_reply or "").strip()
    if um:
        _save_embedding_point(
            user_id, um, "user", "dialogue_snippet", sort_ts, event_utc_iso
        )
    if br:
        _save_embedding_point(
            user_id, br, "bot", "dialogue_snippet", sort_ts + 1e-6, event_utc_iso
        )
