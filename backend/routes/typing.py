"""
routes/typing.py — NeuroType AI Typing Session Routes
HTTP endpoints for receiving keystroke streams and session data from the client,
and for querying aggregate analytics per user.
"""

import logging
from fastapi import APIRouter, HTTPException, Query

from models.schemas import KeystrokePayload, SessionPayload
from services import keystroke_service, analytics_service
from services.ml_model import update_weights
from utils.feature_extractor import extract_features

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Typing"])


# ── POST /typing/keystroke ────────────────────────────────────────────────────

@router.post("/keystroke", summary="Submit a single keystroke event")
def submit_keystroke(payload: KeystrokePayload):
    """
    Records a single keystroke event from the client's real-time stream.

    Use this endpoint for low-latency, event-driven integrations where the
    client emits one event per keypress rather than batching a full session.

    **Request Body** — `KeystrokePayload`:
    - `user_id`       — owner of the event
    - `session_id`    — session this keystroke belongs to
    - `key`           — key label (e.g. "a", "Space", "Backspace")
    - `timestamp`     — Unix timestamp in milliseconds
    - `is_error`      — `true` if this was a correction/mistake keystroke
    - `hold_duration` — how long the key was held (ms)
    """
    keystroke_service.store_keystroke(payload)
    logger.info(
        "Keystroke received — user='%s' session='%s' key='%s' error=%s",
        payload.user_id, payload.session_id, payload.key, payload.is_error,
    )
    return {"status": "ok", "message": "Keystroke recorded"}


# ── POST /typing/session ──────────────────────────────────────────────────────

@router.post("/session", summary="Submit a complete typing session")
def submit_session(payload: SessionPayload):
    """
    Accepts a full batch of keystrokes representing one complete typing session.

    All keystrokes are stored atomically, then session-level statistics
    (WPM, accuracy, duration) are computed and persisted immediately.

    **Request Body** — `SessionPayload`:
    - `user_id`    — owner of the session
    - `session_id` — unique session identifier (client-generated UUID recommended)
    - `keystrokes` — ordered list of `KeystrokePayload` objects
    """
    if not payload.keystrokes:
        raise HTTPException(status_code=400, detail="Session must contain at least one keystroke.")

    for ks in payload.keystrokes:
        # Ensure session_id on each keystroke matches the session envelope
        ks.session_id = payload.session_id
        ks.user_id    = payload.user_id
        keystroke_service.store_keystroke(ks)

    stats = analytics_service.compute_session_stats(payload.session_id, payload.user_id)

    # ── Online Learning (Fix 1) ────────────────────────────────────────────────────────
    # After each session: extract features from the stored keystrokes,
    # then update model weights using the ground-truth error rate.
    # This makes the Cognitive Brain Model adapt to this specific user over time.
    raw_keystrokes = keystroke_service.get_keystrokes(payload.session_id)
    if len(raw_keystrokes) >= 2:
        features = extract_features(raw_keystrokes)
        actual_error_rate = stats.error_keys / stats.total_keys if stats.total_keys else 0.0
        update_weights(features, actual_error_rate)
        logger.info(
            "Learning triggered — session='%s' actual_error_rate=%.3f",
            payload.session_id, actual_error_rate,
        )

    logger.info(
        "Session stored — session='%s' user='%s' keystrokes=%d WPM=%.1f",
        payload.session_id, payload.user_id, len(payload.keystrokes), stats.wpm,
    )
    return {
        "status": "ok",
        "session_id": payload.session_id,
        "stats": stats.model_dump(),
    }


# ── GET /typing/analytics ─────────────────────────────────────────────────────

@router.get("/analytics", summary="Get aggregate analytics for a user")
def get_analytics(user_id: str = Query(..., description="User ID to retrieve analytics for")):
    """
    Returns aggregate typing analytics across all sessions for the given user.

    **Query Parameters**:
    - `user_id` — the user whose analytics to retrieve

    **Response** — `UserAnalytics`:
    - `total_sessions` — number of completed sessions
    - `avg_wpm`        — mean words per minute across all sessions
    - `avg_accuracy`   — mean accuracy percentage
    - `sessions`       — list of individual `SessionStats` records
    """
    analytics = analytics_service.get_user_stats(user_id)
    if analytics.total_sessions == 0:
        raise HTTPException(status_code=404, detail=f"No session data found for user '{user_id}'.")
    return analytics.model_dump()
