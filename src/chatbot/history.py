"""
Persistent chat history.

Two interchangeable backends behind the same interface:

- PostgresChatHistory — the deployment backend. gorest (the company backend)
  calls this service over HTTP and passes the authenticated user's identity
  (user_id = Casdoor user UUID, taxpayer_id = Moadi UUID); every answered
  question is stored per-user in the shared PostgreSQL database so support can
  see who asked what.
- ChatHistory — SQLite fallback for local/offline use (and automatic fallback
  when PostgreSQL is unreachable, so a live demo never breaks).

Recording is always best-effort: history errors are logged, never raised.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_SQLITE_TABLE = """
CREATE TABLE IF NOT EXISTS chat_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at    TEXT    NOT NULL,
    question    TEXT    NOT NULL,
    answer      TEXT    NOT NULL,
    confidence  INTEGER,
    user_id     TEXT,
    taxpayer_id TEXT,
    session_id  TEXT
)
"""

# Older databases predate the identity columns; add them in place. Must run
# BEFORE the indexes — idx_chat_history_user needs the user_id column.
_SQLITE_MIGRATIONS = (
    "ALTER TABLE chat_history ADD COLUMN user_id TEXT",
    "ALTER TABLE chat_history ADD COLUMN taxpayer_id TEXT",
    "ALTER TABLE chat_history ADD COLUMN session_id TEXT",
)

_SQLITE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_chat_history_asked_at ON chat_history(asked_at)",
    "CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(user_id)",
)

_ROW_KEYS = ("asked_at", "question", "answer", "confidence", "user_id", "taxpayer_id", "session_id")


def _rows_to_dicts(rows) -> list[dict]:
    return [dict(zip(_ROW_KEYS, row, strict=True)) for row in rows]


class ChatHistory:
    """Append-only Q&A log backed by SQLite (local fallback backend)."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute(_SQLITE_TABLE)
        for migration in _SQLITE_MIGRATIONS:
            try:
                conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # column already exists
        for index in _SQLITE_INDEXES:
            conn.execute(index)
        return conn

    def add(
        self,
        question: str,
        answer: str,
        confidence: int | None = None,
        user_id: str | None = None,
        taxpayer_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Record one answered question. Never raises — history is best-effort."""
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO chat_history "
                    "(asked_at, question, answer, confidence, user_id, taxpayer_id, session_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        datetime.now().isoformat(timespec="seconds"),
                        question,
                        answer,
                        confidence,
                        user_id,
                        taxpayer_id,
                        session_id,
                    ),
                )
        except sqlite3.Error as e:
            logger.warning(f"Could not record chat history: {e}")

    def recent(self, limit: int = 20, user_id: str | None = None) -> list[dict]:
        """The most recent entries (optionally one user's), oldest first."""
        where = "WHERE user_id = ?" if user_id else ""
        params: tuple = (user_id, limit) if user_id else (limit,)
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT {', '.join(_ROW_KEYS)} FROM chat_history "
                    f"{where} ORDER BY id DESC LIMIT ?",
                    params,
                ).fetchall()
        except sqlite3.Error as e:
            logger.warning(f"Could not read chat history: {e}")
            return []
        return _rows_to_dicts(reversed(rows))

    def users(self) -> list[dict]:
        """Per-user summary: who asked how many questions, and when last."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT user_id, COUNT(*), MIN(asked_at), MAX(asked_at) "
                    "FROM chat_history GROUP BY user_id ORDER BY MAX(asked_at) DESC"
                ).fetchall()
        except sqlite3.Error as e:
            logger.warning(f"Could not read chat history: {e}")
            return []
        return [
            {"user_id": r[0], "questions": r[1], "first_asked_at": r[2], "last_asked_at": r[3]}
            for r in rows
        ]

    def count(self) -> int:
        try:
            with self._connect() as conn:
                return conn.execute("SELECT COUNT(*) FROM chat_history").fetchone()[0]
        except sqlite3.Error:
            return 0


# Timestamps are stored as TIMESTAMPTZ (UTC) and rendered in local time.
_DISPLAY_TZ = "Asia/Tehran"

_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_history (
    id          BIGSERIAL PRIMARY KEY,
    asked_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    confidence  INTEGER,
    user_id     TEXT,
    taxpayer_id TEXT,
    session_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_chat_history_asked_at ON chat_history(asked_at);
CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(user_id, asked_at);
CREATE INDEX IF NOT EXISTS idx_chat_history_taxpayer ON chat_history(taxpayer_id, asked_at);
"""


class PostgresChatHistory:
    """Append-only Q&A log backed by PostgreSQL (deployment backend).

    One connection is opened lazily and reused; if it drops, the next
    operation reconnects once. Raises on construction only if psycopg is
    missing; connection problems surface on first use via ensure_ready().
    """

    def __init__(self, settings: dict):
        import psycopg  # deferred: optional dependency of the SQLite-only setup

        self._psycopg = psycopg
        self._settings = settings
        self._conn = None

    def _connect(self):
        if self._conn is None or self._conn.closed:
            self._conn = self._psycopg.connect(
                host=self._settings["host"],
                port=self._settings["port"],
                user=self._settings["user"],
                password=self._settings["password"],
                dbname=self._settings["dbname"],
                sslmode=self._settings.get("sslmode", "prefer"),
                connect_timeout=self._settings.get("connect_timeout", 3),
                autocommit=True,
            )
            self._conn.execute(_PG_SCHEMA)
        return self._conn

    def ensure_ready(self) -> None:
        """Connect and create the schema; raises if the DB is unreachable."""
        self._connect()

    def _execute(self, query: str, params: tuple = ()):
        """Run one statement, reconnecting once if the connection went stale."""
        try:
            return self._connect().execute(query, params)
        except self._psycopg.OperationalError:
            self._conn = None
            return self._connect().execute(query, params)

    def add(
        self,
        question: str,
        answer: str,
        confidence: int | None = None,
        user_id: str | None = None,
        taxpayer_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Record one answered question. Never raises — history is best-effort."""
        try:
            self._execute(
                "INSERT INTO chat_history "
                "(question, answer, confidence, user_id, taxpayer_id, session_id) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (question, answer, confidence, user_id, taxpayer_id, session_id),
            )
        except self._psycopg.Error as e:
            logger.warning(f"Could not record chat history in PostgreSQL: {e}")

    def recent(self, limit: int = 20, user_id: str | None = None) -> list[dict]:
        """The most recent entries (optionally one user's), oldest first."""
        where = "WHERE user_id = %s" if user_id else ""
        params: tuple = (user_id, limit) if user_id else (limit,)
        try:
            rows = self._execute(
                f"SELECT to_char(asked_at AT TIME ZONE '{_DISPLAY_TZ}', "
                "'YYYY-MM-DD\"T\"HH24:MI:SS'), question, answer, "
                "confidence, user_id, taxpayer_id, session_id FROM chat_history "
                f"{where} ORDER BY id DESC LIMIT %s",
                params,
            ).fetchall()
        except self._psycopg.Error as e:
            logger.warning(f"Could not read chat history from PostgreSQL: {e}")
            return []
        return _rows_to_dicts(reversed(rows))

    def users(self) -> list[dict]:
        """Per-user summary: who asked how many questions, and when last."""
        try:
            rows = self._execute(
                "SELECT user_id, COUNT(*), "
                f"to_char(MIN(asked_at) AT TIME ZONE '{_DISPLAY_TZ}', 'YYYY-MM-DD\"T\"HH24:MI:SS'), "
                f"to_char(MAX(asked_at) AT TIME ZONE '{_DISPLAY_TZ}', 'YYYY-MM-DD\"T\"HH24:MI:SS') "
                "FROM chat_history GROUP BY user_id ORDER BY MAX(asked_at) DESC"
            ).fetchall()
        except self._psycopg.Error as e:
            logger.warning(f"Could not read chat history from PostgreSQL: {e}")
            return []
        return [
            {"user_id": r[0], "questions": r[1], "first_asked_at": r[2], "last_asked_at": r[3]}
            for r in rows
        ]

    def count(self) -> int:
        try:
            return self._execute("SELECT COUNT(*) FROM chat_history").fetchone()[0]
        except self._psycopg.Error:
            return 0


def create_history(config) -> "PostgresChatHistory | ChatHistory | None":
    """Build the configured history backend.

    HISTORY_BACKEND == "postgres" tries PostgreSQL first and falls back to
    SQLite (with a logged warning) when the server is unreachable, so the
    chatbot keeps working offline.
    """
    if not config.HISTORY_ENABLED:
        return None
    if config.HISTORY_BACKEND == "postgres":
        if not config.HISTORY_PG.get("password"):
            logger.info(
                "No PostgreSQL password configured (set CHATBOT_PG_PASSWORD or "
                "create .env) — using SQLite history."
            )
            return ChatHistory(config.HISTORY_DB)
        try:
            store = PostgresChatHistory(config.HISTORY_PG)
            store.ensure_ready()
            return store
        except Exception as e:  # noqa: BLE001 - any failure means "use the fallback"
            logger.warning(
                f"PostgreSQL history unavailable ({e}) — falling back to SQLite "
                f"at {config.HISTORY_DB}"
            )
    return ChatHistory(config.HISTORY_DB)
