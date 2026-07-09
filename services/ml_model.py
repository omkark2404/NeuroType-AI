"""
services/ml_model.py — NeuroType AI Cognitive Typing Brain Model
Lightweight sigmoid-based model. Predicts fatigue, error probability, and
typing consistency from behavioral features extracted by feature_extractor.py.

No heavy ML frameworks are required — this is a hand-tuned, interpretable
model that runs in pure Python/NumPy and produces real-time predictions.

Model Architecture:
    Fatigue     = sigmoid( W_VAR * variance_interval + W_HOLD * avg_hold_time )
    Error Prob  = sigmoid( W_ERR * error_rate + W_FATIGUE * fatigue )
    Consistency = 100 - std_dev(intervals)   [clamped to 0-100]
"""

import logging
import math
import threading
from typing import Any, Dict

from config import settings
from models.schemas import PredictionResult

logger = logging.getLogger(__name__)


# ── Mutable Global Weights (Learning Layer) ────────────────────────────────────
# These start from config defaults and drift over time via update_weights().
# In-process only (reset on restart); persist to DB for production continuity.
_weights: Dict[str, float] = {
    "variance_interval": settings.WEIGHT_VARIANCE_INTERVAL,
    "avg_hold_time":     settings.WEIGHT_AVG_HOLD_TIME,
    "error_rate":        settings.WEIGHT_ERROR_RATE,
    "fatigue_feedback":  settings.WEIGHT_FATIGUE_FEEDBACK,
}
_weights_lock = threading.Lock()


# ── Activation Function ───────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """
    Standard logistic sigmoid: maps any real number to (0, 1).

    Used as the activation function for both the fatigue and error-probability
    sub-models so outputs are always valid probability-like scalars.

    Args:
        x: Raw linear combination of weighted features.

    Returns:
        Float in the open interval (0, 1).
    """
    return 1.0 / (1.0 + math.exp(-x))


def _std_dev(values: list) -> float:
    """Population standard deviation over a list of floats."""
    if not values:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


# ── Core Model ────────────────────────────────────────────────────────────────

def predict_behavior(features: Dict[str, Any]) -> PredictionResult:
    """
    Cognitive Typing Brain Model - converts behavioral features into
    three cognitive-state predictions.

    FATIGUE MODEL:
      Fatigue rises when the typist's rhythm becomes erratic
      (high variance_interval) and when keystrokes become sluggish
      (high avg_hold_time). Both inputs are linearly weighted before
      being squashed through sigmoid -> output in (0, 1).

    ERROR PROBABILITY MODEL:
      Error likelihood is driven by the raw error rate PLUS the current
      fatigue level (fatigued users make more errors). Sigmoid keeps
      the output as a probability in (0, 1).

    CONSISTENCY SCORE:
      A perfectly consistent typist would have zero variance in their
      inter-keystroke intervals. Score = 100 - std_dev(intervals),
      clamped to [0, 100]. Higher = more consistent rhythm.

    Args:
        features: Dict produced by feature_extractor.extract_features():
            avg_interval      (float, ms)
            variance_interval (float, ms^2)
            avg_hold_time     (float, ms)
            error_rate        (float, 0-1)
            burst_speed       (float, keys/sec)
            intervals_list    (List[float])

    Returns:
        PredictionResult with fatigue, error_prob, consistency fields.
    """
    # Normalise raw ms values into a model-friendly range so the sigmoid
    # input stays in (-3, +3) for typical human typing sessions.
    #
    # variance_interval is in ms^2 - typical range 1,000-20,000 ms^2
    #   -> divide by 10,000 -> range 0.1-2.0
    # avg_hold_time is in ms - typical range 40-300 ms
    #   -> divide by 300 -> range 0.13-1.0
    # error_rate is already 0-1
    norm_variance  = features["variance_interval"] / 10_000.0
    norm_hold_time = features["avg_hold_time"]      / 300.0
    norm_error     = features["error_rate"]

    # ── Fatigue ───────────────────────────────────────────────────────────
    with _weights_lock:
        fatigue_input = (
            _weights["variance_interval"] * norm_variance
            + _weights["avg_hold_time"]   * norm_hold_time
        )
    fatigue = _sigmoid(fatigue_input)

    # ── Error Probability ─────────────────────────────────────────────────
    with _weights_lock:
        error_input = (
            _weights["error_rate"]        * norm_error
            + _weights["fatigue_feedback"] * fatigue
        )
    error_prob = _sigmoid(error_input)

    # ── Consistency Score ─────────────────────────────────────────────────
    intervals = features.get("intervals_list", [])
    std = _std_dev(intervals) if intervals else 0.0
    # Normalise std_dev similarly (divide by 100) so score reflects
    # perceptible rhythm irregularity rather than raw millisecond spread.
    consistency = max(0.0, min(100.0, 100.0 - (std / 100.0) * 10.0))

    result = PredictionResult(
        fatigue=round(fatigue, 4),
        error_prob=round(error_prob, 4),
        consistency=round(consistency, 2),
    )

    logger.info(
        "CognitiveBrainModel -- fatigue=%.3f | error_prob=%.3f | consistency=%.1f",
        result.fatigue, result.error_prob, result.consistency,
    )
    return result


# ── Online Learning Layer ─────────────────────────────────────────────────────

def update_weights(features: Dict[str, Any], actual_error_rate: float) -> None:
    """
    Online gradient descent - adjusts model weights after each session
    using the ground-truth error rate as the supervision signal.

    Algorithm (single-step SGD on the error-prob sub-model):
        predicted  = current model's error_prob output
        error      = actual_error_rate - predicted      (signed residual)
        delta_w_i  = LEARNING_RATE * error * feature_i  (gradient step)
        w_i        = clamp(w_i + delta_w_i, 0.01, 2.0)  (prevent explosion)

    The clamp keeps every weight in a numerically safe range regardless of
    how many sessions accumulate - no weight can go negative or explode.

    IMPORTANT - Deadlock prevention:
        predict_behavior() acquires _weights_lock internally.
        update_weights() must NOT call predict_behavior() while holding
        _weights_lock (threading.Lock is non-reentrant; same-thread
        re-acquisition deadlocks forever). We resolve this by computing
        ALL predictions in a single call BEFORE entering the lock.

    Args:
        features:          Feature dict from feature_extractor.extract_features().
        actual_error_rate: Ground-truth error rate (error_keys / total_keys)
                           measured from the completed session.
    """
    global _weights

    # Single predict_behavior call OUTSIDE the lock.
    # Captures both error_prob (for residual) and fatigue (for fatigue_feedback
    # weight update) without any risk of lock re-entry deadlock.
    current_pred    = predict_behavior(features)
    predicted       = current_pred.error_prob
    current_fatigue = current_pred.fatigue
    error = actual_error_rate - predicted

    # Features and their corresponding weight keys
    feature_to_weight = {
        "variance_interval": "variance_interval",
        "avg_hold_time":     "avg_hold_time",
        "error_rate":        "error_rate",
    }

    with _weights_lock:
        for feat_key, weight_key in feature_to_weight.items():
            feature_val = features.get(feat_key, 0.0)
            # Normalise feature to the same scale used in predict_behavior
            if feat_key == "variance_interval":
                feature_val /= 10_000.0
            elif feat_key == "avg_hold_time":
                feature_val /= 300.0
            # error_rate is already 0-1

            delta = settings.LEARNING_RATE * error * feature_val
            _weights[weight_key] = max(0.01, min(2.0, _weights[weight_key] + delta))

        # fatigue_feedback weight uses the precomputed fatigue as its "feature"
        delta_ff = settings.LEARNING_RATE * error * current_fatigue
        _weights["fatigue_feedback"] = max(0.01, min(2.0, _weights["fatigue_feedback"] + delta_ff))

    logger.info(
        "WeightUpdate -- residual=%.4f | w_var=%.4f w_hold=%.4f w_err=%.4f w_fat=%.4f",
        error,
        _weights["variance_interval"],
        _weights["avg_hold_time"],
        _weights["error_rate"],
        _weights["fatigue_feedback"],
    )


def get_current_weights() -> Dict[str, float]:
    """
    Returns a snapshot of the current learned weights.
    Useful for debugging, admin endpoints, or persisting weights.
    """
    with _weights_lock:
        return dict(_weights)
