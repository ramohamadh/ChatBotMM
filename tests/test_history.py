"""Tests for the persistent chat history (SQLite backend + backend factory)."""

import types

from chatbot.history import ChatHistory, create_history


def test_add_and_recent(tmp_path):
    store = ChatHistory(tmp_path / "history.db")
    store.add("سوال اول", "جواب اول", confidence=90)
    store.add("سوال دوم", "جواب دوم", confidence=None)

    entries = store.recent()
    assert len(entries) == 2
    # recent() returns oldest first
    assert entries[0]["question"] == "سوال اول"
    assert entries[0]["confidence"] == 90
    assert entries[1]["answer"] == "جواب دوم"
    assert entries[1]["confidence"] is None
    assert store.count() == 2


def test_recent_limit(tmp_path):
    store = ChatHistory(tmp_path / "history.db")
    for i in range(5):
        store.add(f"q{i}", f"a{i}")
    entries = store.recent(limit=2)
    assert [e["question"] for e in entries] == ["q3", "q4"]


def test_persists_across_instances(tmp_path):
    db = tmp_path / "history.db"
    ChatHistory(db).add("سوال", "جواب")
    assert ChatHistory(db).count() == 1


def test_creates_parent_directory(tmp_path):
    db = tmp_path / "nested" / "dir" / "history.db"
    store = ChatHistory(db)
    store.add("q", "a")
    assert db.exists()


def test_user_identity_and_filtering(tmp_path):
    store = ChatHistory(tmp_path / "history.db")
    store.add("q1", "a1", user_id="user-a", taxpayer_id="tp-1", session_id="s1")
    store.add("q2", "a2", user_id="user-b")
    store.add("q3", "a3", user_id="user-a")

    entries = store.recent(user_id="user-a")
    assert [e["question"] for e in entries] == ["q1", "q3"]
    assert entries[0]["taxpayer_id"] == "tp-1"
    assert entries[0]["session_id"] == "s1"

    summary = {s["user_id"]: s["questions"] for s in store.users()}
    assert summary == {"user-a": 2, "user-b": 1}


def test_migrates_old_schema(tmp_path):
    """A pre-identity-columns database gains the new columns in place."""
    import sqlite3

    db = tmp_path / "history.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "asked_at TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL, "
            "confidence INTEGER)"
        )
        conn.execute(
            "INSERT INTO chat_history (asked_at, question, answer) VALUES ('t', 'q', 'a')"
        )

    store = ChatHistory(db)
    store.add("q2", "a2", user_id="user-a")
    entries = store.recent()
    assert len(entries) == 2
    assert entries[0]["user_id"] is None
    assert entries[1]["user_id"] == "user-a"


def _config(tmp_path, backend):
    return types.SimpleNamespace(
        HISTORY_ENABLED=True,
        HISTORY_BACKEND=backend,
        HISTORY_DB=tmp_path / "history.db",
        HISTORY_PG={
            # Unroutable address: forces the postgres branch to fail fast.
            "host": "127.0.0.1",
            "port": 1,
            "user": "x",
            "password": "x",
            "dbname": "x",
            "sslmode": "disable",
            "connect_timeout": 1,
        },
    )


def test_factory_sqlite_backend(tmp_path):
    store = create_history(_config(tmp_path, "sqlite"))
    assert isinstance(store, ChatHistory)


def test_factory_falls_back_to_sqlite_when_pg_unreachable(tmp_path):
    store = create_history(_config(tmp_path, "postgres"))
    assert isinstance(store, ChatHistory)


def test_factory_disabled():
    config = types.SimpleNamespace(HISTORY_ENABLED=False)
    assert create_history(config) is None
