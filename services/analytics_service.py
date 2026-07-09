"""
services/analytics_service.py — NeuroType AI Analytics Service
Computes WPM, accuracy, and session-level statistics from raw keystroke data,
then persists and retrieves aggregate user analytics.
"""

import logging
from typing import Any, Dict, List

from services import keystroke_service
from models import storage
from models.schemas import SessionStats, UserAnalytics

logger = logging.getLogger(__name__)

# Standard typing convention: 1 word ≈ 5 keystrokes
_CHARS_PER_WORD = 5


def compute_session_stats(session_id: str, user_id: str) -> SessionStats:
    """
    Computes and persists WPM, accuracy, and duration for a session.

    Algorithm:
        total_keys    = len(keystrokes)
        error_keys    = count of keystrokes where is_error == True
        accuracy      = (total_keys - error_keys) / total_keys * 100
        duration_min  = (last_timestamp - first_timestamp) / 60_000   [ms → min]
        wpm           = (total_keys / 5) / duration_min

    Special cases:
        - If total_keys == 0, all stats are set to 0.
        - If duration_min == 0 (single-ms session), wpm is clamped to 0.

    Args:
        session_id: Unique session identifier.
        user_id:    Owner of the session (stored in the stats record).

    Returns:
        SessionStats Pydantic model with all computed fields.
    """
    keystrokes = keystroke_service.get_keystrokes(session_id)
    total_keys = len(keystrokes)

    if total_keys == 0:
        logger.warning("No keystrokes found for session='%s' — returning zero stats", session_id)
        stats = SessionStats(
            session_id=session_id,
            user_id=user_id,
            total_keys=0,
            error_keys=0,
            accuracy=0.0,
            wpm=0.0,
            duration_min=0.0,
        )
        storage.save_session_stats(stats.model_dump())
        return stats

    error_keys = sum(1 for k in keystrokes if bool(k.get("is_error", False)))
    accuracy = ((total_keys - error_keys) / total_keys) * 100.0

    start_ts = keystrokes[0]["timestamp"]
    end_ts   = keystrokes[-1]["timestamp"]
    duration_ms  = max(end_ts - start_ts, 1)        # prevent division by zero
    duration_min = duration_ms / 60_000.0
    words_typed  = total_keys / _CHARS_PER_WORD
    wpm          = words_typed / duration_min if duration_min > 0 else 0.0

    stats = SessionStats(
        session_id=session_id,
        user_id=user_id,
        total_keys=total_keys,
        error_keys=error_keys,
        accuracy=round(accuracy, 2),
        wpm=round(wpm, 2),
        duration_min=round(duration_min, 4),
    )

    storage.save_session_stats(stats.model_dump())
    logger.info(
        "Session stats computed — session='%s' WPM=%.1f Accuracy=%.1f%% Keys=%d",
        session_id, wpm, accuracy, total_keys,
    )
    return stats


def get_user_stats(user_id: str) -> UserAnalytics:
    """
    Retrieves and aggregates stats across all sessions for a given user.

    Args:
        user_id: The user whose analytics to retrieve.

    Returns:
        UserAnalytics containing totals, averages, and the full session list.
    """
    raw_sessions = storage.get_sessions_for_user(user_id)

    if not raw_sessions:
        logger.info("No sessions found for user='%s'", user_id)
        return UserAnalytics(
            user_id=user_id,
            total_sessions=0,
            avg_wpm=0.0,
            avg_accuracy=0.0,
            sessions=[],
        )

    sessions = [SessionStats(**s) for s in raw_sessions]
    avg_wpm      = sum(s.wpm      for s in sessions) / len(sessions)
    avg_accuracy = sum(s.accuracy for s in sessions) / len(sessions)

    logger.info(
        "User analytics — user='%s' sessions=%d avg_wpm=%.1f avg_accuracy=%.1f%%",
        user_id, len(sessions), avg_wpm, avg_accuracy,
    )
    trend = compute_trend(user_id)
    return UserAnalytics(
        user_id=user_id,
        total_sessions=len(sessions),
        avg_wpm=round(avg_wpm, 2),
        avg_accuracy=round(avg_accuracy, 2),
        trend=trend,
        sessions=sessions,
    )


# ── Trend Analysis (Fix 2) ─────────────────────────────────────────────────────────────

def compute_trend(user_id: str) -> str:
    """
    Analyses the WPM slope across the last 10 sessions to determine whether
    the user’s performance is trending upward, downward, or holding steady.

    Method: ordinary least-squares linear regression on WPM values ordered
    oldest → newest. No NumPy required — computed with stdlib arithmetic.

    Args:
        user_id: The user to analyse.

    Returns:
        One of: "improving" | "declining" | "stable" | "insufficient_data"
    """
    raw = storage.get_last_n_sessions(user_id, n=10)
    if len(raw) < 3:
        logger.info("Trend: insufficient data for user='%s' (sessions=%d)", user_id, len(raw))
        return "insufficient_data"

    wpm_values = [float(s.get("wpm", 0.0)) for s in raw]
    slope = _least_squares_slope(wpm_values)

    if slope > 0.5:
        trend = "improving"
    elif slope < -0.5:
        trend = "declining"
    else:
        trend = "stable"

    logger.info(
        "Trend for user='%s': %s (slope=%.3f over %d sessions)",
        user_id, trend, slope, len(wpm_values),
    )
    return trend


def _least_squares_slope(values: List[float]) -> float:
    """
    Computes the least-squares linear regression slope over a list of values.

    x = [0, 1, 2, ..., n−1]  (session indices)
    y = values                (WPM per session)

    slope = ( n⋗Σ(xy) − Σx⋗Σy ) / ( n⋗Σ(x²) − (Σx)² )

    Returns 0.0 when the denominator is zero (all x values identical, i.e. n=1).
    """
    n = len(values)
    x = list(range(n))
    sum_x  = sum(x)
    sum_y  = sum(values)
    sum_xy = sum(xi * yi for xi, yi in zip(x, values))
    sum_x2 = sum(xi * xi for xi in x)
    denom  = n * sum_x2 - sum_x ** 2
    if denom == 0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom


# ── Weak Pattern Detection (Fix 4) ──────────────────────────────────────────────────────

def detect_weak_patterns(session_id: str) -> List[str]:
    """
    Identifies the top-3 bigrams (2-character sequences) where the user’s
    errors cluster, enabling personalised drill targets.

    A bigram is formed from the key immediately before an error keystroke
    and the error key itself (e.g. pressing ‘h’ incorrectly after ‘t’ → "th").

    Args:
        session_id: The session to analyse.

    Returns:
        List of up to 3 bigram strings sorted by error frequency (descending).
        Empty list if no errors were recorded.
    """
    keystrokes = keystroke_service.get_keystrokes(session_id)
    bigram_errors: Dict[str, int] = {}

    for i in range(1, len(keystrokes)):
        if bool(keystrokes[i].get("is_error", False)):
            prev_key = keystrokes[i - 1].get("key", "?")
            curr_key = keystrokes[i].get("key", "?")
            # Skip non-character keys to keep bigrams readable
            if len(prev_key) == 1 and len(curr_key) == 1:
                bigram = prev_key + curr_key
                bigram_errors[bigram] = bigram_errors.get(bigram, 0) + 1

    if not bigram_errors:
        return []

    sorted_bigrams = sorted(bigram_errors.items(), key=lambda item: item[1], reverse=True)
    top_3 = [bigram for bigram, _ in sorted_bigrams[:3]]
    logger.info("WeakPatterns for session='%s': %s", session_id, top_3)
    return top_3
