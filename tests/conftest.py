"""Shared fixtures: keep heavy stacks (embedder, Gemini) out of default imports."""

import pytest


@pytest.fixture(autouse=True)
def _no_dotenv_override(monkeypatch):
    """Tests set env explicitly; avoid .env on disk masking expectations."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
