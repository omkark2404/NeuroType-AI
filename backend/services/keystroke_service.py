"""
services/keystroke_service.py — NeuroType AI Keystroke Storage Service
Accepts individual keystroke events and persists them to the storage layer.
All reads are ordered by timestamp ascending to preserve stream integrity.
"""

import logging
from typing import Any, Dict, List

from models import storage
from models.schemas import KeystrokePayload
from config import settings

logger = logging.getLogger(__name__)


def store_keystroke(payload: KeystrokePayload) -> None:
    """
    Persists a single keystroke event to the database.

    Each record stores the full context needed for feature extraction:
    user_id, session_id, key label, Unix-ms timestamp, error flag,
    and hold duration.

    Args:
        payload: Validated KeystrokePayload from the API layer.
    """
    record = {
        "user_id":       payload.user_id,
        "session_id":    payload.session_id,
        "key":           payload.key,
        "timestamp":     payload.timestamp,
        "is_error":      int(payload.is_error),   # SQLite stores booleans as 0/1
        "hold_duration": payload.hold_duration,
    }
    storage.get_db().insert("keystrokes", record)
    logger.debug(
        "Keystroke stored — key='%s' user='%s' session='%s' error=%s",
        payload.key, payload.user_id, payload.session_id, payload.is_error,
    )


def get_keystrokes(session_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves all keystroke records for a given session, ordered by
    timestamp ascending so feature extraction always sees a correct stream.

    Args:
        session_id: The session identifier to query.

    Returns:
        List of keystroke dicts, each with keys:
        user_id, session_id, key, timestamp, is_error, hold_duration.
    """
    db = storage.get_db()

    if settings.DB_TYPE == "sqlite":
        rows = db.query(
            "SELECT * FROM keystrokes WHERE session_id = ? ORDER BY timestamp ASC",
            [session_id],
        )
    else:
        # MemoryDB: filter by session_id, then sort in Python
        rows = db.query("keystrokes", [("session_id", session_id)])
        rows = sorted(rows, key=lambda r: r["timestamp"])

    # Normalise is_error back to bool regardless of backend
    for row in rows:
        row["is_error"] = bool(row.get("is_error", 0))

    logger.debug(
        "Fetched %d keystrokes for session='%s'", len(rows), session_id
    )
    return rows
