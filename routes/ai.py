"""
routes/ai.py — NeuroType AI Cognitive Prediction & Adaptation Routes
Exposes the real-time cognitive inference pipeline and adaptive difficulty engine
over HTTP. These are the core "intelligence" endpoints of the system.
"""

import logging
from fastapi import APIRouter, HTTPException

from config import settings
from models.schemas import PredictRequest, AdaptRequest, AdaptResponse, StreamPredictRequest
from services import keystroke_service, ai_engine, adaptive_engine, analytics_service
from utils.cache import cache

logger = logging.getLogger(__name__)
router = APIRouter(tags=["AI"])


# ── POST /ai/predict ──────────────────────────────────────────────────────────

@router.post("/predict", response_model=dict, summary="Run cognitive model prediction")
def predict(request: PredictRequest):
    """
    Runs the full cognitive inference pipeline on a stored session's keystroke
    stream and returns fatigue, error probability, and consistency scores.

    **Pipeline**:
    `keystrokes` → `feature_extractor` → `ml_model` → `PredictionResult`

    **Request Body** — `PredictRequest`:
    - `user_id`    — identifies the user (for logging/audit)
    - `session_id` — the session whose keystrokes to analyse

    **Response**:
    ```json
    {
        "fatigue_level":     0.42,
        "error_probability": 0.31,
        "consistency_score": 78.5
    }
    ```
    - `fatigue_level`     ∈ (0, 1) — 0 = fresh, 1 = exhausted
    - `error_probability` ∈ (0, 1) — predicted likelihood of making an error
    - `consistency_score` ∈ (0, 100) — rhythmic regularity of typing
    """
    keystrokes = keystroke_service.get_keystrokes(request.session_id)

    if not keystrokes:
        raise HTTPException(
            status_code=404,
            detail=f"No keystroke data found for session '{request.session_id}'.",
        )

    # ── Cache check (Fix 5) ────────────────────────────────────────────────────────
    cache_key = f"predict:{request.session_id}"
    cached = cache.get(cache_key)
    if cached:
        logger.info("Cache HIT — predict session='%s'", request.session_id)
        return cached

    try:
        predictions = ai_engine.run_prediction(keystrokes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    logger.info(
        "Prediction served — user='%s' session='%s' fatigue=%.3f error_prob=%.3f",
        request.user_id, request.session_id,
        predictions.fatigue, predictions.error_prob,
    )
    result = {
        "fatigue_level":     predictions.fatigue,
        "error_probability": predictions.error_prob,
        "consistency_score": predictions.consistency,
    }
    cache.set(cache_key, result, ttl=settings.CACHE_TTL_SECONDS)
    return result


# ── POST /ai/adapt ────────────────────────────────────────────────────────────

@router.post("/adapt", response_model=AdaptResponse, summary="Get difficulty directive + coaching feedback")
def adapt(request: AdaptRequest):
    """
    Runs the full cognitive pipeline **and** the adaptive difficulty engine
    in one call. Returns a difficulty directive, human-readable coaching
    feedback, and the underlying predictions.

    **Pipeline**:
    `keystrokes` → `feature_extractor` → `ml_model` → `adaptive_engine` → response

    **Request Body** — `AdaptRequest`:
    - `user_id`    — identifies the user
    - `session_id` — session to analyse

    **Response** — `AdaptResponse`:
    ```json
    {
        "directive": "focus_accuracy_exercises",
        "feedback":  "Your error rate is climbing. Focus on accuracy over speed.",
        "predictions": {
            "fatigue":     0.42,
            "error_prob":  0.61,
            "consistency": 73.4
        }
    }
    ```

    **Possible directives**:
    | Directive                  | Condition                          |
    |----------------------------|------------------------------------|
    | `reduce_difficulty`        | fatigue > 0.7                      |
    | `focus_accuracy_exercises` | error_prob > 0.5                   |
    | `increase_difficulty`      | consistency > 80.0                 |
    | `maintain_difficulty`      | all metrics within normal range    |
    """
    keystrokes = keystroke_service.get_keystrokes(request.session_id)

    if not keystrokes:
        raise HTTPException(
            status_code=404,
            detail=f"No keystroke data found for session '{request.session_id}'.",
        )

    try:
        predictions = ai_engine.run_prediction(keystrokes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    directive = adaptive_engine.adapt_difficulty(predictions)
    feedback  = adaptive_engine.generate_feedback(predictions)
    weak_patterns = analytics_service.detect_weak_patterns(request.session_id)

    logger.info(
        "Adapt served — user='%s' session='%s' directive='%s' weak_patterns=%s",
        request.user_id, request.session_id, directive, weak_patterns,
    )
    return AdaptResponse(
        directive=directive,
        feedback=feedback,
        weak_patterns=weak_patterns,
        predictions=predictions,
    )


# ── POST /ai/stream-predict (Fix 3) ──────────────────────────────────────────────────────

@router.post("/stream-predict", response_model=dict, summary="Real-time inference on live keystroke stream")
def stream_predict(request: StreamPredictRequest):
    """
    Runs cognitive inference **directly on a raw keystroke list** sent in the
    request body — no session storage round-trip required.

    This is the true real-time endpoint. The client sends the last N keystrokes
    as they arrive and receives predictions + a directive immediately.

    **Request Body** — `StreamPredictRequest`:
    - `user_id`    — identifies the user
    - `keystrokes` — list of `KeystrokePayload` objects (minimum 2)

    **Response**:
    ```json
    {
        "fatigue":      0.38,
        "error_prob":   0.44,
        "consistency":  81.2,
        "directive":    "increase_difficulty",
        "feedback":     "Excellent consistency! You're ready to increase your pace.",
        "mode":         "realtime"
    }
    ```
    """
    if len(request.keystrokes) < 2:
        raise HTTPException(
            status_code=400,
            detail="stream-predict requires at least 2 keystrokes for inference.",
        )

    # Cache key based on user + timestamp of the most recent keystroke
    last_ts   = request.keystrokes[-1].timestamp
    cache_key = f"stream:{request.user_id}:{last_ts}"
    cached    = cache.get(cache_key)
    if cached:
        logger.info("Cache HIT — stream-predict user='%s' ts=%d", request.user_id, last_ts)
        return cached

    # Convert Pydantic models to dicts for the inference pipeline
    ks_dicts = [ks.model_dump() for ks in request.keystrokes]

    try:
        predictions = ai_engine.run_prediction(ks_dicts)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    directive = adaptive_engine.adapt_difficulty(predictions)
    feedback  = adaptive_engine.generate_feedback(predictions)

    result = {
        "fatigue":     predictions.fatigue,
        "error_prob":  predictions.error_prob,
        "consistency": predictions.consistency,
        "directive":   directive,
        "feedback":    feedback,
        "mode":        "realtime",
    }
    cache.set(cache_key, result, ttl=settings.CACHE_TTL_SECONDS)

    logger.info(
        "StreamPredict — user='%s' keystrokes=%d fatigue=%.3f directive='%s'",
        request.user_id, len(request.keystrokes), predictions.fatigue, directive,
    )
    return result
