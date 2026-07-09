"""
routes/auth.py — NeuroType AI Authentication Routes
Simple JWT-based authentication to associate typing sessions with users.
Passwords are hashed with bcrypt. Tokens are signed with HS256.
"""

import logging
from fastapi import APIRouter, HTTPException

from models.schemas import RegisterPayload, LoginPayload, TokenResponse
from models import storage
from utils.helpers import hash_password, verify_password, create_token

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Auth"])


# ── POST /auth/register ───────────────────────────────────────────────────────

@router.post("/register", summary="Register a new user account")
def register(payload: RegisterPayload):
    """
    Creates a new user account. Passwords are bcrypt-hashed before storage.

    **Request Body** — `RegisterPayload`:
    - `username` — 3–50 characters, must be unique
    - `password` — minimum 6 characters

    Returns a confirmation message. Use `/auth/login` next to obtain a JWT token.
    """
    if storage.get_user(payload.username):
        raise HTTPException(
            status_code=400,
            detail=f"Username '{payload.username}' is already taken.",
        )

    hashed = hash_password(payload.password)
    storage.create_user(payload.username, hashed)
    logger.info("User registered — username='%s'", payload.username)
    return {"status": "ok", "message": f"User '{payload.username}' registered successfully."}


# ── POST /auth/login ──────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse, summary="Login and receive a JWT token")
def login(payload: LoginPayload):
    """
    Authenticates a user and returns a signed JWT access token.

    **Request Body** — `LoginPayload`:
    - `username` — registered username
    - `password` — account password

    **Response** — `TokenResponse`:
    - `access_token` — signed JWT (HS256), valid for 24 hours by default
    - `token_type`   — always `"bearer"`

    Include the token in subsequent requests as:
    `Authorization: Bearer <access_token>`
    """
    user = storage.get_user(payload.username)

    if not user or not verify_password(payload.password, user["hashed_password"]):
        logger.warning("Failed login attempt for username='%s'", payload.username)
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = create_token({"sub": payload.username})
    logger.info("User authenticated — username='%s'", payload.username)
    return TokenResponse(access_token=token, token_type="bearer")
