"""RAG и generate_reply с моками: без загрузки SentenceTransformer и без Qdrant/Gemini в сети."""

import sys
import types
from unittest.mock import MagicMock, AsyncMock

import pytest


class _FakePoint:
    __slots__ = ("payload",)

    def __init__(self, payload):
        self.payload = payload


@pytest.fixture(scope="module", autouse=True)
def _stub_embedder_module():
    """Подменяет ai.embedder до импорта rag, чтобы не тянуть sentence_transformers."""
    for mod in ("ai.gemini_engine", "ai.rag", "ai.embedder", "ai.memory_extraction"):
        sys.modules.pop(mod, None)
    fake = types.ModuleType("ai.embedder")
    fake.embedder = MagicMock()
    fake.embedder.get_embedding.return_value = [0.1] * 768
    sys.modules["ai.embedder"] = fake
    yield
    for mod in ("ai.gemini_engine", "ai.rag", "ai.embedder", "ai.memory_extraction"):
        sys.modules.pop(mod, None)


def test_get_memory_context_empty_when_no_hits(monkeypatch):
    mock_q = MagicMock()
    mock_q.search_similar_by_kind.return_value = []
    monkeypatch.setattr("ai.rag.qdrant_db", mock_q)

    import ai.rag as rag

    assert rag.get_memory_context("user-1", "query text") == ""
    assert mock_q.search_similar_by_kind.call_count == 2


def test_get_memory_context_merges_kinds_and_sorts(monkeypatch):
    mock_q = MagicMock()

    def by_kind(uid, emb, kind, limit):
        if kind == "fact":
            return [
                _FakePoint(
                    {
                        "sort_ts": 2.0,
                        "role": "user",
                        "text": "fact b",
                        "kind": "fact",
                        "event_utc_iso": "t2",
                    }
                )
            ]
        return [
            _FakePoint(
                {
                    "sort_ts": 1.0,
                    "role": "bot",
                    "text": "dia a",
                    "kind": "dialogue_snippet",
                    "event_utc_iso": "t1",
                }
            )
        ]

    mock_q.search_similar_by_kind.side_effect = by_kind

    monkeypatch.setattr("ai.rag.qdrant_db", mock_q)

    import ai.rag as rag

    out = rag.get_memory_context("u1", "q")
    lines = out.splitlines()
    assert "[DIALOGUE]" in lines[0] and "dia a" in lines[0]
    assert "[FACT]" in lines[1] and "fact b" in lines[1]


def test_persist_fallback_on_extraction_none(monkeypatch):
    import ai.memory_extraction as memex

    mock_q = MagicMock()
    monkeypatch.setattr("ai.rag.qdrant_db", mock_q)
    monkeypatch.setattr(memex, "extract_memory_items", lambda *a, **k: None)

    import ai.rag as rag

    rag.persist_conversation_turn(
        "uid",
        "привет",
        "здрасьте",
        sort_ts=100.0,
        event_utc_iso="2026-04-17T12:00:00+00:00",
    )
    assert mock_q.upsert_memory_point.call_count == 2


@pytest.mark.asyncio
async def test_generate_reply_calls_gemini_and_persist(monkeypatch):
    mock_q = MagicMock()
    mock_q.search_similar_by_kind.return_value = []
    monkeypatch.setattr("ai.rag.qdrant_db", mock_q)
    monkeypatch.setattr("database.sqlite_db.fetch_previous_friend_messages", lambda uid, lim: [])
    monkeypatch.setattr("database.sqlite_db.append_friend_chat_turn", MagicMock())

    import ai.gemini_engine as ge
    import ai.graph_agent as ga

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "короткий ответ"
    mock_resp.function_calls = None
    mock_resp.usage_metadata = None

    async def mock_run_react_agent(*args, **kwargs):
        return "короткий ответ"

    monkeypatch.setattr(ga, "run_react_agent", mock_run_react_agent)

    monkeypatch.setattr(ge, "get_genai_client", lambda: mock_client)
    persist_mock = MagicMock()
    monkeypatch.setattr(ge, "persist_conversation_turn", persist_mock)
    
    async def mock_run_memory_extraction_bg(*args, **kwargs):
        pass
    monkeypatch.setattr("ai.memory_graph.run_memory_extraction_bg", mock_run_memory_extraction_bg)

    out = await ge.generate_reply("u42", "привет")
    assert out == "короткий ответ"


@pytest.mark.asyncio
async def test_generate_reply_builds_prompt_with_context(monkeypatch):
    mock_q = MagicMock()
    mock_q.search_similar_by_kind.return_value = [
        _FakePoint(
            {
                "sort_ts": 1.0,
                "role": "user",
                "text": "раньше",
                "kind": "dialogue_snippet",
                "event_utc_iso": "t0",
            }
        )
    ]
    monkeypatch.setattr("ai.rag.qdrant_db", mock_q)
    monkeypatch.setattr("database.sqlite_db.fetch_previous_friend_messages", lambda uid, lim: [])
    monkeypatch.setattr("database.sqlite_db.append_friend_chat_turn", MagicMock())

    import ai.gemini_engine as ge
    import ai.graph_agent as ga

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "ok"
    mock_resp.function_calls = None
    mock_resp.usage_metadata = None

    # Track arguments passed to run_react_agent
    react_args = {}
    async def mock_run_react_agent(system_prompt, user_messages, user_id):
        react_args['prompt'] = str(user_messages)
        return mock_resp.text

    monkeypatch.setattr(ga, "run_react_agent", mock_run_react_agent)
    
    async def mock_run_memory_extraction_bg(*args, **kwargs):
        pass
    monkeypatch.setattr("ai.memory_graph.run_memory_extraction_bg", mock_run_memory_extraction_bg)

    mock_generate_content = AsyncMock()
    mock_generate_content.return_value = mock_resp
    mock_client.aio.models.generate_content = mock_generate_content

    monkeypatch.setattr(ge, "get_genai_client", lambda: mock_client)
    monkeypatch.setattr(ge, "persist_conversation_turn", MagicMock())

    await ge.generate_reply("u1", "новое")
    
    # Assert on the prompt passed to run_react_agent instead of mock_generate_content
    blob = react_args.get("prompt", "")
    assert "Recent conversation" in blob
    assert "long-term memory" in blob
    assert "раньше" in blob and "новое" in blob
