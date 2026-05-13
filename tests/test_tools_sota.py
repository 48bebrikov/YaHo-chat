import pytest
from ai.tools_sota import execute_python_code

def test_execute_python_code_simple():
    code = "print('hello world')"
    res = execute_python_code(code)
    assert res.strip() == "hello world"

def test_execute_python_code_variable():
    # Tests that when there's no print, it grabs the last variable
    code = "a = 2 + 2\nx = 10"
    res = execute_python_code(code)
    assert "x = 10" in res

def test_execute_python_code_error():
    # Tests exception capturing
    code = "1 / 0"
    res = execute_python_code(code)
    assert "Error executing code: ZeroDivisionError:" in res

def test_add_user_reminder(monkeypatch):
    from unittest.mock import MagicMock
    import database.sqlite_db as sqlite_db
    
    mock_db = MagicMock()
    
    class DummyContext:
        def __enter__(self):
            return mock_db
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(sqlite_db, "db_session", DummyContext)
    
    from ai.tools_sota import add_user_reminder
    
    # Call the tool
    res = add_user_reminder("u1", "feed the dog", 15)
    
    assert "Reminder successfully set" in res
    assert "15 minutes" in res
    
    # Verify DB add was called
    mock_db.add.assert_called_once()
    added_reminder = mock_db.add.call_args[0][0]
    assert added_reminder.user_id == "u1"
    assert added_reminder.text == "feed the dog"
    assert added_reminder.is_sent == 0
