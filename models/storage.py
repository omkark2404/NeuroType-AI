"""
models/storage.py — NeuroType AI Storage Abstraction Layer
Abstracts all database operations. Swappable between SQLite and in-memory dict
without changing any service code — all services call get_db() and work against
a consistent DBInterface regardless of backend.
"""

import sqlite3
import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)


# ── Abstract Interface ─────────────────────────────────────────────────────────

class DBInterface(ABC):
    """Abstract database interface. All backends must implement these methods."""

    @abstractmethod
    def insert(self, table: str, record: Dict[str, Any]) -> None:
        """Insert a new record into the given table/store."""
        ...

    @abstractmethod
    def query(self, sql_or_key: str, params: Optional[List] = None) -> List[Dict]:
        """Execute a query and return matching records as dicts."""
        ...

    @abstractmethod
    def get_one(self, table: str, filters: Dict[str, Any]) -> Optional[Dict]:
        """Retrieve a single record matching all filter key/values."""
        ...

    @abstractmethod
    def upsert(self, table: str, record: Dict[str, Any], key_field: str) -> None:
        """Insert or replace a record identified by key_field."""
        ...


# ── In-Memory Backend ──────────────────────────────────────────────────────────

class MemoryDB(DBInterface):
    """
    Thread-safe in-memory storage backend.
    Data persists only for the lifetime of the process.
    Useful for development, testing, and demo deployments.
    """

    _lock = threading.Lock()
    _store: Dict[str, List[Dict]] = {
        "keystrokes": [],
        "session_stats": [],
        "users": [],
    }

    def insert(self, table: str, record: Dict[str, Any]) -> None:
        with self._lock:
            self._store.setdefault(table, []).append(dict(record))
        logger.debug("MemoryDB.insert → table=%s", table)

    def query(self, sql_or_key: str, params: Optional[List] = None) -> List[Dict]:
        """
        For MemoryDB, sql_or_key is interpreted as a table name.
        Optional params: list of (field, value) tuples used as equality filters.
        """
        with self._lock:
            rows = list(self._store.get(sql_or_key, []))
        if params:
            for field, value in params:
                rows = [r for r in rows if r.get(field) == value]
        return rows

    def get_one(self, table: str, filters: Dict[str, Any]) -> Optional[Dict]:
        with self._lock:
            rows = self._store.get(table, [])
        for row in rows:
            if all(row.get(k) == v for k, v in filters.items()):
                return dict(row)
        return None

    def upsert(self, table: str, record: Dict[str, Any], key_field: str) -> None:
        with self._lock:
            rows = self._store.setdefault(table, [])
            for i, row in enumerate(rows):
                if row.get(key_field) == record.get(key_field):
                    rows[i] = dict(record)
                    return
            rows.append(dict(record))


# ── SQLite Backend ─────────────────────────────────────────────────────────────

class SQLiteDB(DBInterface):
    """
    SQLite-backed persistent storage.
    Uses row_factory=sqlite3.Row so all results are returned as dicts.
    Creates a new connection per call (safe for multithreaded FastAPI workers).
    """

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(settings.SQLITE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _row_to_dict(self, row) -> Dict:
        return dict(row) if row else {}

    def insert(self, table: str, record: Dict[str, Any]) -> None:
        cols = ", ".join(record.keys())
        placeholders = ", ".join("?" * len(record))
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        with self._connect() as conn:
            conn.execute(sql, list(record.values()))
            conn.commit()
        logger.debug("SQLiteDB.insert → table=%s", table)

    def query(self, sql_or_key: str, params: Optional[List] = None) -> List[Dict]:
        """sql_or_key is a full SQL SELECT statement for SQLite."""
        with self._connect() as conn:
            cur = conn.execute(sql_or_key, params or [])
            return [self._row_to_dict(r) for r in cur.fetchall()]

    def get_one(self, table: str, filters: Dict[str, Any]) -> Optional[Dict]:
        clauses = " AND ".join(f"{k} = ?" for k in filters)
        sql = f"SELECT * FROM {table} WHERE {clauses} LIMIT 1"
        with self._connect() as conn:
            cur = conn.execute(sql, list(filters.values()))
            row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def upsert(self, table: str, record: Dict[str, Any], key_field: str) -> None:
        cols = ", ".join(record.keys())
        placeholders = ", ".join("?" * len(record))
        sql = f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})"
        with self._connect() as conn:
            conn.execute(sql, list(record.values()))
            conn.commit()


# ── Initialization ─────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Bootstraps the storage backend.
    SQLite: creates all required tables if they do not already exist.
    Memory: no-op — store is pre-initialized at import time.
    """
    if settings.DB_TYPE == "sqlite":
        conn = sqlite3.connect(settings.SQLITE_PATH)
        cur = conn.cursor()

        # Enable WAL mode: allows concurrent reads while a write is in progress.
        # This prevents the lock-contention timeouts that occur under concurrent load.
        cur.execute("PRAGMA journal_mode=WAL;")
        # Set a 5-second busy timeout so concurrent writers wait instead of failing.
        cur.execute("PRAGMA busy_timeout=5000;")

        cur.executescript("""
            CREATE TABLE IF NOT EXISTS keystrokes (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       TEXT    NOT NULL,
                session_id    TEXT    NOT NULL,
                key           TEXT    NOT NULL,
                timestamp     INTEGER NOT NULL,
                is_error      INTEGER NOT NULL DEFAULT 0,
                hold_duration REAL    NOT NULL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS session_stats (
                session_id   TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL,
                total_keys   INTEGER,
                error_keys   INTEGER,
                accuracy     REAL,
                wpm          REAL,
                duration_min REAL
            );

            CREATE TABLE IF NOT EXISTS users (
                username        TEXT PRIMARY KEY,
                hashed_password TEXT NOT NULL
            );
        """)
        conn.commit()
        conn.close()
        logger.info("SQLite DB initialized at '%s' (WAL mode enabled)", settings.SQLITE_PATH)
    else:
        logger.info("In-memory storage backend initialized")


def get_db() -> DBInterface:
    """
    Factory function. Returns the correct DBInterface implementation
    based on the DB_TYPE setting. Services call this and never import
    the concrete classes directly.
    """
    if settings.DB_TYPE == "sqlite":
        return SQLiteDB()
    return MemoryDB()


# ── High-Level Helpers (used by services) ────────────────────────────────────

def save_session_stats(stats: Dict[str, Any]) -> None:
    """Persist (or update) computed session statistics."""
    get_db().upsert("session_stats", stats, key_field="session_id")


def get_sessions_for_user(user_id: str) -> List[Dict]:
    """Retrieve all session stat records for a given user_id."""
    db = get_db()
    if settings.DB_TYPE == "sqlite":
        return db.query(
            "SELECT * FROM session_stats WHERE user_id = ?", [user_id]
        )
    return db.query("session_stats", [("user_id", user_id)])


def get_user(username: str) -> Optional[Dict]:
    """Look up a user record by username."""
    return get_db().get_one("users", {"username": username})


def create_user(username: str, hashed_password: str) -> None:
    """Create a new user record in storage."""
    get_db().insert("users", {"username": username, "hashed_password": hashed_password})


def get_last_n_sessions(user_id: str, n: int = 10) -> List[Dict]:
    """
    Returns the N most-recent session stat records for a user, ordered
    oldest → newest so trend analysis sees correct chronological order.

    Args:
        user_id: The user to query.
        n:       Maximum number of sessions to return (default 10).

    Returns:
        List of session stat dicts, length ≤ n.
    """
    db = get_db()
    if settings.DB_TYPE == "sqlite":
        # Use rowid as a proxy for insertion order (monotonically increasing)
        rows = db.query(
            "SELECT * FROM session_stats WHERE user_id = ? "
            "ORDER BY rowid DESC LIMIT ?",
            [user_id, n],
        )
        # Reverse so the list goes oldest → newest for slope calculation
        return list(reversed(rows))
    else:
        all_rows = db.query("session_stats", [("user_id", user_id)])
        return all_rows[-n:]   # MemoryDB preserves insertion order
