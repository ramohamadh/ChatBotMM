"""Shared test configuration.

Force HuggingFace offline mode: every model a test needs is either stubbed or
already in the local cache, and the hub's "is there a newer revision?" check
can hang the whole suite on a slow or saturated connection.

Force the SQLite history backend into a temp location: tests must NEVER write
chat history to the shared company PostgreSQL database.
"""

import os
import tempfile

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

os.environ["CHATBOT_HISTORY_BACKEND"] = "sqlite"
os.environ["CHATBOT_HISTORY_DB"] = os.path.join(
    tempfile.mkdtemp(prefix="chatbot-test-"), "history.db"
)

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_history_singleton():
    """get_chat_history() caches its store; reset it between tests."""
    from chatbot import commands

    commands._history_store = None
    commands._history_loaded = False
    yield
    commands._history_store = None
    commands._history_loaded = False
