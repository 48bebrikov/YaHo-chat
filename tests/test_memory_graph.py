import pytest
from unittest.mock import MagicMock
from ai.memory_graph import analyze_dialogue, MemoryState

def test_analyze_dialogue_json_extraction(monkeypatch):
    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = '```json\n{"facts": ["User likes coffee", "User has a dog"], "topic": "morning routine"}\n```'
    mock_llm.invoke.return_value = mock_resp

    monkeypatch.setattr("ai.memory_graph.get_memory_llm", lambda: mock_llm)

    state = MemoryState(
        user_id="1", 
        user_message="Good morning! Drinking coffee with my dog.", 
        bot_reply="Sounds great!",
        extracted_facts=[],
        topic=""
    )

    res = analyze_dialogue(state)
    
    assert res["topic"] == "morning routine"
    assert len(res["extracted_facts"]) == 2
    assert "User likes coffee" in res["extracted_facts"]

def test_analyze_dialogue_error_handling(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("API overloaded")

    monkeypatch.setattr("ai.memory_graph.get_memory_llm", lambda: mock_llm)

    state = MemoryState(
        user_id="1", 
        user_message="Hello", 
        bot_reply="Hi",
        extracted_facts=[],
        topic=""
    )

    res = analyze_dialogue(state)
    
    # Should safely return fallback empty data
    assert res["extracted_facts"] == []
    assert res["topic"] == "general chat"
