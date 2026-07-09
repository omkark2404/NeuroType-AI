"""
app.py — NeuroType AI Application Entry Point
Boots the FastAPI application, registers all routers, configures middleware,
and initializes the storage backend on startup.

Run with:
    uvicorn app:app --reload
"""

import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from config import settings
from utils.helpers import setup_logging
from utils.cache import init_cache
from models.storage import init_db
from routes.typing import router as typing_router
from routes.ai import router as ai_router
from routes.auth import router as auth_router

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle for the NeuroType AI engine."""
    # Startup
    setup_logging()
    init_db()
    init_cache()
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  NeuroType AI — Cognitive Typing Intelligence Engine")
    logger.info("  Version  : 1.1.0")
    logger.info("  Storage  : %s", settings.DB_TYPE.upper())
    logger.info("  Host     : %s:%d", settings.HOST, settings.PORT)
    logger.info("  Docs     : http://%s:%d/docs", settings.HOST, settings.PORT)
    logger.info("=" * 60)
    yield
    # Shutdown
    logger.info("NeuroType AI engine shutting down.")


# ── App Factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Constructs and configures the FastAPI application instance.

    Responsibilities:
      - Attach CORS middleware (open in dev; restrict origins in production)
      - Register all route modules under their canonical prefixes
      - Wire the lifespan context for startup/shutdown hooks

    Returns:
        Configured FastAPI instance ready to be served by Uvicorn.
    """
    application = FastAPI(
        title="NeuroType AI",
        version="1.1.0",
        description=(
            "## NeuroType AI — Cognitive Typing Intelligence Engine\n\n"
            "A **real-time cognitive behavior modeling system** for human-computer interaction.\n\n"
            "### Core capabilities\n"
            "- 🧠 **Behavioral Feature Extraction** — inter-keystroke intervals, hold times, burst speed\n"
            "- 📊 **Cognitive Brain Model** — sigmoid-based fatigue, error-probability, consistency\n"
            "- 🎯 **Adaptive Difficulty Engine** — dynamic difficulty directives + coaching feedback\n"
            "- 🔄 **Online Learning** — model weights adapt after every session via gradient descent\n"
            "- ⚡ **Real-time Stream Inference** — `/ai/stream-predict` bypasses storage entirely\n"
            "- 📈 **Trend Analysis** — WPM slope across last 10 sessions (improving/declining/stable)\n"
            "- 🎯 **Weak Pattern Detection** — personalised bigram drill targets from error clusters\n"
            "- ⚡ **TTL Cache** — prediction results cached 30s to avoid redundant recomputes\n"
            "- 🔐 **JWT Authentication** — secure user sessions\n\n"
            "Explore the endpoints below or hit `/docs` for the interactive Swagger UI."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],        # Restrict to specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    application.include_router(auth_router,   prefix="/auth")
    application.include_router(typing_router, prefix="/typing")
    application.include_router(ai_router,     prefix="/ai")

    # ── Static Files (Frontend) ───────────────────────────────────────────────
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        application.mount("/static", StaticFiles(directory=static_dir), name="static")

    return application


app = create_app()


# ── Root — Serve Frontend UI ──────────────────────────────────────────────────

@app.get("/", tags=["Health"], summary="NeuroType AI frontend or health check")
def root():
    """
    Serves the interactive NeuroType AI web frontend if static/index.html exists.
    Falls back to a JSON health-check response (useful for API-only deployments).
    """
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {
        "service": "NeuroType AI — Cognitive Typing Intelligence Engine",
        "version": "1.1.0",
        "status":  "operational",
        "docs":    "/docs",
        "frontend": "not available — static/index.html not found",
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
