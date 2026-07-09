"""
config.py — NeuroType AI Configuration & Constants
Centralizes all environment variables, ML model weights, thresholds, and app settings.
"""

import logging
import os


class Settings:
    """Application-wide settings and cognitive model hyperparameters."""

    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # ── Storage ───────────────────────────────────────────────────────────────
    DB_TYPE: str = os.getenv("DB_TYPE", "sqlite")       # "sqlite" or "memory"
    SQLITE_PATH: str = os.getenv("SQLITE_PATH", "neurotype.db")

    # ── JWT Auth ──────────────────────────────────────────────────────────────
    SECRET_KEY: str = os.getenv("SECRET_KEY", "neurotype-secret-change-in-production")
    ALGORITHM: str = "HS256"
    TOKEN_TTL_MINUTES: int = int(os.getenv("TOKEN_TTL_MINUTES", "1440"))   # 24 hours

    # ── ML Model Weights (hand-tuned sigmoid inputs) ──────────────────────────
    # These weights control how strongly each feature influences the prediction.
    # Increasing WEIGHT_VARIANCE_INTERVAL makes the fatigue model more sensitive
    # to irregular typing rhythm.
    WEIGHT_VARIANCE_INTERVAL: float = float(os.getenv("W_VAR_INTERVAL", "0.4"))
    WEIGHT_AVG_HOLD_TIME: float = float(os.getenv("W_HOLD_TIME", "0.3"))
    WEIGHT_ERROR_RATE: float = float(os.getenv("W_ERROR_RATE", "0.5"))
    WEIGHT_FATIGUE_FEEDBACK: float = float(os.getenv("W_FATIGUE_FEEDBACK", "0.6"))

    # ── Adaptive Engine Thresholds ────────────────────────────────────────────
    FATIGUE_HIGH_THRESHOLD: float = float(os.getenv("FATIGUE_THRESHOLD", "0.7"))
    ERROR_PROB_HIGH_THRESHOLD: float = float(os.getenv("ERROR_THRESHOLD", "0.5"))
    CONSISTENCY_HIGH_THRESHOLD: float = float(os.getenv("CONSISTENCY_THRESHOLD", "80.0"))

    # ── Online Learning ───────────────────────────────────────────────────────
    # Gradient descent step size for update_weights().
    # Smaller = more stable but slower to adapt; larger = faster but noisier.
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.05"))

    # ── Cache ─────────────────────────────────────────────────────────────────
    # Time-to-live (seconds) for prediction cache entries.
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "30"))

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "[%(asctime)s] %(levelname)s [%(name)s] — %(message)s"


settings = Settings()
