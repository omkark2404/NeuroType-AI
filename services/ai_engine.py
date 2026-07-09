"""
services/ai_engine.py — NeuroType AI Orchestration Engine
Single entry point for all AI logic. Wires the feature extraction pipeline
directly into the cognitive ML model and returns structured predictions.

Pipeline:
    List[KeystrokeRecord]
        → feature_extractor.extract_features()
        → ml_model.predict_behavior()
        → PredictionResult
"""

import logging
from typing import Any, Dict, List

from utils import feature_extractor
from services import ml_model
from models.schemas import PredictionResult

logger = logging.getLogger(__name__)


def run_prediction(keystrokes: List[Dict[str, Any]]) -> PredictionResult:
    """
    Full cognitive inference pipeline.

    Accepts a raw keystroke stream (as returned from storage), extracts
    behavioral features, feeds them into the Cognitive Typing Brain Model,
    and returns a structured PredictionResult.

    Args:
        keystrokes: List of keystroke records (dicts) retrieved from storage.
                    Each record must contain: timestamp, is_error, hold_duration.

    Returns:
        PredictionResult containing:
            fatigue     ∈ (0, 1)   — higher = more fatigued
            error_prob  ∈ (0, 1)   — higher = more error-prone
            consistency ∈ (0, 100) — higher = more consistent rhythm

    Raises:
        ValueError: if the keystroke list is empty.
    """
    n = len(keystrokes)
    logger.info("AIEngine — running cognitive prediction on %d keystrokes", n)

    if n == 0:
        raise ValueError("Cannot run prediction on an empty keystroke list.")

    # Step 1 — Behavioral feature extraction
    features = feature_extractor.extract_features(keystrokes)
    logger.debug(
        "AIEngine — features: avg_interval=%.1f ms | variance=%.1f | "
        "error_rate=%.3f | burst=%.2f keys/s",
        features["avg_interval"],
        features["variance_interval"],
        features["error_rate"],
        features["burst_speed"],
    )

    # Step 2 — Cognitive model inference
    predictions = ml_model.predict_behavior(features)
    logger.info(
        "AIEngine — predictions: fatigue=%.3f | error_prob=%.3f | consistency=%.1f",
        predictions.fatigue,
        predictions.error_prob,
        predictions.consistency,
    )

    return predictions
