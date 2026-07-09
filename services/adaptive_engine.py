"""
services/adaptive_engine.py — NeuroType AI Adaptive Difficulty Engine
Translates cognitive model predictions into concrete difficulty directives
and human-readable coaching feedback.

Decision logic is a priority-ordered threshold tree:
    1. HIGH fatigue   → reduce difficulty  (protect the user first)
    2. HIGH error_prob → focus accuracy mode
    3. HIGH consistency → increase difficulty (user is ready)
    4. Default        → maintain current difficulty
"""

import logging
from models.schemas import PredictionResult
from config import settings

logger = logging.getLogger(__name__)


# ── Directives ────────────────────────────────────────────────────────────────

DIRECTIVE_REDUCE   = "reduce_difficulty"
DIRECTIVE_ACCURACY = "focus_accuracy_exercises"
DIRECTIVE_INCREASE = "increase_difficulty"
DIRECTIVE_MAINTAIN = "maintain_difficulty"


# ── Public API ────────────────────────────────────────────────────────────────

def adapt_difficulty(predictions: PredictionResult) -> str:
    """
    Decides the difficulty adjustment directive based on the user's current
    cognitive state as predicted by ml_model.predict_behavior().

    Priority order (first matching condition wins):
        1. fatigue    > FATIGUE_HIGH_THRESHOLD      → reduce_difficulty
        2. error_prob > ERROR_PROB_HIGH_THRESHOLD   → focus_accuracy_exercises
        3. consistency > CONSISTENCY_HIGH_THRESHOLD → increase_difficulty
        4. else                                      → maintain_difficulty

    Args:
        predictions: PredictionResult from the cognitive brain model.

    Returns:
        A directive string that the client/game engine should act on.
    """
    if predictions.fatigue > settings.FATIGUE_HIGH_THRESHOLD:
        logger.info(
            "AdaptiveEngine → REDUCE | fatigue=%.3f (threshold=%.2f)",
            predictions.fatigue, settings.FATIGUE_HIGH_THRESHOLD,
        )
        return DIRECTIVE_REDUCE

    if predictions.error_prob > settings.ERROR_PROB_HIGH_THRESHOLD:
        logger.info(
            "AdaptiveEngine → ACCURACY_FOCUS | error_prob=%.3f (threshold=%.2f)",
            predictions.error_prob, settings.ERROR_PROB_HIGH_THRESHOLD,
        )
        return DIRECTIVE_ACCURACY

    if predictions.consistency > settings.CONSISTENCY_HIGH_THRESHOLD:
        logger.info(
            "AdaptiveEngine → INCREASE | consistency=%.1f (threshold=%.1f)",
            predictions.consistency, settings.CONSISTENCY_HIGH_THRESHOLD,
        )
        return DIRECTIVE_INCREASE

    logger.info("AdaptiveEngine → MAINTAIN")
    return DIRECTIVE_MAINTAIN


def generate_feedback(predictions: PredictionResult) -> str:
    """
    Generates natural-language coaching feedback for the user based on
    their current cognitive state. Mirrors the priority order of adapt_difficulty()
    so that directive and feedback are always coherent.

    Args:
        predictions: PredictionResult from the cognitive brain model.

    Returns:
        A plain-English coaching message suitable for display to the user.
    """
    if predictions.fatigue > settings.FATIGUE_HIGH_THRESHOLD:
        return (
            "You're showing signs of cognitive fatigue. "
            "Take a short break before your next session."
        )

    if predictions.error_prob > settings.ERROR_PROB_HIGH_THRESHOLD:
        return (
            "Your error rate is climbing. "
            "Focus on accuracy over speed — slow down and type deliberately."
        )

    if predictions.consistency > settings.CONSISTENCY_HIGH_THRESHOLD:
        return (
            "Excellent rhythm and consistency! "
            "You're ready to challenge yourself — increase your pace."
        )

    return (
        "Good performance. "
        "Maintain your current rhythm and keep building momentum."
    )
