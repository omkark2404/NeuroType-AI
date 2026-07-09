"""
utils/feature_extractor.py — NeuroType AI Behavioral Feature Extractor
Transforms raw keystroke streams into ML-ready numerical feature vectors.

Pipeline:
    List[KeystrokeRecord] → sort by timestamp
                          → compute inter-keystroke intervals
                          → compute hold times
                          → compute error rate
                          → compute burst speed
                          → return feature dict
"""

import logging
import math
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_features(keystrokes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Converts a raw keystroke stream into a behavioral feature vector.

    Args:
        keystrokes: List of keystroke records (dicts with keys:
                    user_id, session_id, key, timestamp, is_error, hold_duration)

    Returns:
        dict with keys:
            avg_interval      (float, ms)        — mean time between keystrokes
            variance_interval (float, ms²)       — variance of inter-keystroke intervals
            avg_hold_time     (float, ms)         — mean key hold duration
            error_rate        (float, 0.0–1.0)   — fraction of keystrokes that are errors
            burst_speed       (float, keys/sec)  — peak local typing speed
            intervals_list    (List[float])       — raw intervals for std-dev calculation
    """
    if len(keystrokes) < 2:
        logger.warning("Insufficient keystrokes (%d) for feature extraction — returning zeros", len(keystrokes))
        return _default_zero_features()

    # Sort ascending by timestamp so intervals are always positive
    sorted_ks = sorted(keystrokes, key=lambda k: k["timestamp"])

    # ── Inter-keystroke intervals ──────────────────────────────────────────────
    intervals: List[float] = [
        float(sorted_ks[i]["timestamp"] - sorted_ks[i - 1]["timestamp"])
        for i in range(1, len(sorted_ks))
    ]

    # Guard against zero-length intervals (same-ms events)
    intervals = [max(i, 0.1) for i in intervals]

    avg_interval = _mean(intervals)
    variance_interval = _variance(intervals, avg_interval)

    # ── Hold times ────────────────────────────────────────────────────────────
    hold_times: List[float] = [float(k.get("hold_duration", 0.0)) for k in sorted_ks]
    avg_hold_time = _mean(hold_times)

    # ── Error rate ────────────────────────────────────────────────────────────
    errors = sum(1 for k in sorted_ks if bool(k.get("is_error", False)))
    error_rate = errors / len(sorted_ks)

    # ── Burst speed ───────────────────────────────────────────────────────────
    burst_speed = _compute_burst_speed(sorted_ks, window_ms=2000)

    features = {
        "avg_interval": round(avg_interval, 4),
        "variance_interval": round(variance_interval, 4),
        "avg_hold_time": round(avg_hold_time, 4),
        "error_rate": round(error_rate, 4),
        "burst_speed": round(burst_speed, 4),
        "intervals_list": intervals,
    }

    logger.debug(
        "Features extracted — avg_interval=%.1f ms, variance=%.1f, "
        "error_rate=%.3f, burst=%.2f keys/s",
        avg_interval, variance_interval, error_rate, burst_speed,
    )
    return features


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _compute_burst_speed(keystrokes: List[Dict], window_ms: int = 2000) -> float:
    """
    Sliding-window burst speed detector.

    For each keystroke i, counts how many keystrokes fall within the next
    `window_ms` milliseconds, then converts that count to keys/sec.
    Returns the maximum observed rate across all windows.

    Args:
        keystrokes: Sorted (ascending timestamp) keystroke records.
        window_ms:  Width of the detection window in milliseconds.

    Returns:
        Peak typing rate in keystrokes per second.
    """
    max_burst = 0.0
    n = len(keystrokes)
    for i in range(n):
        t_start = keystrokes[i]["timestamp"]
        count = sum(
            1 for k in keystrokes[i:]
            if 0 <= k["timestamp"] - t_start <= window_ms
        )
        rate = count / (window_ms / 1000.0)
        if rate > max_burst:
            max_burst = rate
    return max_burst


def _mean(values: List[float]) -> float:
    """Returns the arithmetic mean, or 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


def _variance(values: List[float], mean: float) -> float:
    """Returns population variance, or 0.0 for lists shorter than 2."""
    if len(values) < 2:
        return 0.0
    return sum((v - mean) ** 2 for v in values) / len(values)


def _std_dev(values: List[float]) -> float:
    """Returns population standard deviation."""
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(_variance(values, mean))


def _default_zero_features() -> Dict[str, Any]:
    """Returns a zeroed feature vector used when there is insufficient data."""
    return {
        "avg_interval": 0.0,
        "variance_interval": 0.0,
        "avg_hold_time": 0.0,
        "error_rate": 0.0,
        "burst_speed": 0.0,
        "intervals_list": [],
    }
