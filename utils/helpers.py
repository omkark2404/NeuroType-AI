"""
utils/helpers.py — NeuroType AI Utility Functions
Auth helpers (JWT creation/verification, password hashing) and shared utilities.
"""

import logging
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from jose import JWTError, jwt

from config import settings

logger = logging.getLogger(__name__)

# ── Password Hashing ──────────────────────────────────────────────────────────


def hash_password(plain: str) -> str:
    """
    Returns a bcrypt hash of the plaintext password.

    Args:
        plain: Plaintext password string.

    Returns:
        Bcrypt-hashed password string safe to store in the database.
    """
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """
    Verifies a plaintext password against a stored bcrypt hash.

    Args:
        plain:  Plaintext password submitted by the user.
        hashed: Stored bcrypt hash from the database.

    Returns:
        True if the password matches, False otherwise.
    """
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ── JWT Tokens ────────────────────────────────────────────────────────────────

def create_token(data: Dict[str, Any]) -> str:
    """
    Creates a signed JWT with an expiry timestamp.

    The payload includes all fields from `data` plus an `exp` claim
    set to now + TOKEN_TTL_MINUTES from settings.

    Args:
        data: Dict of claims to embed (typically {"sub": username}).

    Returns:
        Encoded JWT string.
    """
    payload = dict(data)
    expire = datetime.now(tz=timezone.utc) + timedelta(minutes=settings.TOKEN_TTL_MINUTES)
    payload["exp"] = expire
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    logger.debug("JWT created for subject='%s'", data.get("sub"))
    return token


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decodes and validates a JWT token.

    Args:
        token: Encoded JWT string.

    Returns:
        Decoded payload dict, or None if the token is invalid/expired.
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError as exc:
        logger.warning("JWT decode failed: %s", exc)
        return None


# ── Logging Setup ─────────────────────────────────────────────────────────────

def setup_logging() -> None:
    """
    Configures application-wide structured logging using settings from config.py.
    Should be called once at application startup before any other log statements.
    """
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
        format=settings.LOG_FORMAT,
    )
    logger.info("NeuroType AI logging initialized — level=%s", settings.LOG_LEVEL)
