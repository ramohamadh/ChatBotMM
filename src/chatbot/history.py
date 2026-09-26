"""
Persistent chat history, stored in a local SQLite database.

Every answered question is recorded (question, answer, confidence, timestamp)
so past conversations survive across sessions — `/history` in the chat and
`chatbot history` on the command line read from here. SQLite ships with
Python, so this adds no dependency and works fully offline.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at    TEXT    NOT NULL,
    question    TEXT    NOT NULL,
    answer      TEXT    NOT NULL,
    confidence  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chat_history_asked_at ON chat_history(asked_at);
"""


class ChatHistory:
    """Append-only Q&A log backed by SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.executescript(_SCHEMA)
        return conn

    def add(self, question: str, answer: str, confidence: int | None = None) -> None:
        """Record one answered question. Never raises — history is best-effort."""
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO chat_history (asked_at, question, answer, confidence) "
                    "VALUES (?, ?, ?, ?)",
                    (datetime.now().isoformat(timespec="seconds"), question, answer, confidence),
                )
        except sqlite3.Error as e:
            logger.warning(f"Could not record chat history: {e}")

    def recent(self, limit: int = 20) -> list[dict]:
        """Return the most recent entries, oldest first."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT asked_at, question, answer, confidence FROM chat_history "
                    "ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        except sqlite3.Error as e:
            logger.warning(f"Could not read chat history: {e}")
            return []
        return [
            {"asked_at": r[0], "question": r[1], "answer": r[2], "confidence": r[3]}
            for r in reversed(rows)
        ]

    def count(self) -> int:
        try:
            with self._connect() as conn:
                return conn.execute("SELECT COUNT(*) FROM chat_history").fetchone()[0]
        except sqlite3.Error:
            return 0
