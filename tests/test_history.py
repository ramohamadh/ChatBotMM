"""Tests for the persistent chat history (SQLite)."""

from chatbot.history import ChatHistory


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
