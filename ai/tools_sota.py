import io
import contextlib
from typing import Dict, Any
from datetime import datetime, timezone, timedelta

def execute_python_code(code: str) -> str:
    """
    Executes Python code and returns the stdout output.
    Uses contextlib.redirect_stdout to capture print statements.
    
    Warning: In a real production environment, this should run in an isolated 
    container/sandbox. For YaHo 2026, we will use a basic eval/exec with stdout capture.
    """
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            # Use empty globals and locals for a modicum of isolation, 
            # though it's still dangerous in a real public bot
            local_vars = {}
            exec(code, {"__builtins__": __builtins__}, local_vars)
            
        output = stdout.getvalue()
        if not output and local_vars:
            # If nothing was printed but variables were set, try to return the last one
            last_key = list(local_vars.keys())[-1]
            output = f"{last_key} = {local_vars[last_key]}"
            
        return output if output else "Code executed successfully but returned no output."
    except Exception as e:
        return f"Error executing code: {type(e).__name__}: {str(e)}"

def add_user_reminder(user_id: str, text: str, delay_minutes: int) -> str:
    """Adds a reminder to the SQLite database."""
    from database.sqlite_db import db_session, Reminder
    try:
        remind_time = datetime.now(timezone.utc) + timedelta(minutes=delay_minutes)
        with db_session() as db:
            rem = Reminder(
                user_id=user_id,
                text=text,
                remind_at=remind_time,
                is_sent=0
            )
            db.add(rem)
        return f"Reminder successfully set for {delay_minutes} minutes from now."
    except Exception as e:
        return f"Failed to set reminder: {e}"
