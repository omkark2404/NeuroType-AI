"""
models/schemas.py — NeuroType AI Pydantic Schemas
Defines all request/response data shapes for input validation and API documentation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ── Keystroke / Session ────────────────────────────────────────────────────────

class KeystrokePayload(BaseModel):
    """A single keystroke event emitted by the client."""
    user_id: str = Field(..., description="Unique identifier of the user")
    session_id: str = Field(..., description="Session this keystroke belongs to")
    key: str = Field(..., description="The key that was pressed (single char or label)")
    timestamp: int = Field(..., description="Unix timestamp in milliseconds")
    is_error: bool = Field(False, description="Whether this keystroke was a correction/error")
    hold_duration: float = Field(0.0, description="How long the key was held down, in ms")


class SessionPayload(BaseModel):
    """A complete typing session containing a stream of keystrokes."""
    user_id: str = Field(..., description="Unique identifier of the user")
    session_id: str = Field(..., description="Unique session identifier")
    keystrokes: List[KeystrokePayload] = Field(..., description="Ordered list of keystroke events")


# ── AI Request / Response ──────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Request to run the cognitive prediction model on a session."""
    user_id: str
    session_id: str


class AdaptRequest(BaseModel):
    """Request to get difficulty adaptation directive + coaching feedback."""
    user_id: str
    session_id: str


class StreamPredictRequest(BaseModel):
    """Request to run real-time inference on a raw keystroke stream (no DB round-trip)."""
    user_id: str
    keystrokes: List[KeystrokePayload] = Field(
        ..., min_length=2, description="Live keystroke stream (minimum 2 events)"
    )


class PredictionResult(BaseModel):
    """Output of the Cognitive Typing Brain Model."""
    fatigue: float = Field(..., ge=0.0, le=1.0, description="Fatigue level (0 = fresh, 1 = exhausted)")
    error_prob: float = Field(..., ge=0.0, le=1.0, description="Predicted error probability")
    consistency: float = Field(..., ge=0.0, le=100.0, description="Typing rhythm consistency score")


class AdaptResponse(BaseModel):
    """Full adaptive engine response with directive, feedback, weak patterns, and predictions."""
    directive: str = Field(..., description="Difficulty directive from the adaptive engine")
    feedback: str = Field(..., description="Human-readable coaching feedback")
    weak_patterns: List[str] = Field(default_factory=list, description="Top bigrams where errors cluster")
    predictions: PredictionResult


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterPayload(BaseModel):
    """User registration payload."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class LoginPayload(BaseModel):
    """User login payload."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """JWT token returned on successful login."""
    access_token: str
    token_type: str = "bearer"


# ── Analytics ─────────────────────────────────────────────────────────────────

class SessionStats(BaseModel):
    """Computed statistics for a single typing session."""
    session_id: str
    user_id: str
    total_keys: int
    error_keys: int
    accuracy: float = Field(..., description="Accuracy percentage (0–100)")
    wpm: float = Field(..., description="Words per minute")
    duration_min: float = Field(..., description="Session duration in minutes")


class UserAnalytics(BaseModel):
    """Aggregate analytics across all sessions for a user."""
    user_id: str
    total_sessions: int
    avg_wpm: float
    avg_accuracy: float
    trend: str = Field("insufficient_data", description="Performance trend: improving | declining | stable | insufficient_data")
    sessions: List[SessionStats]
