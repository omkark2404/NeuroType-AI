"""
utils/cache.py — NeuroType AI TTL Cache
Simple in-memory key-value cache with per-entry TTL expiry.
Thread-safe for single-process FastAPI deployments.

Wired into:
  - POST /ai/predict      (cache prediction results by session_id)
  - POST /ai/stream-predict (cache by user_id + last keystroke timestamp)
"""

import time
import threading
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TTLCache:
    """
    In-memory key-value cache with per-entry TTL expiry.

    Each entry stores:
        value      — the cached object
        expires_at — Unix timestamp after which the entry is stale

    Thread safety: a single RLock guards all mutations.
    """

    def __init__(self) -> None:
        self._store: dict = {}          # { key: {"value": any, "expires_at": float} }
        self._lock = threading.RLock()

    def set(self, key: str, value: Any, ttl: int) -> None:
        """
        Store a value under key with an expiry of now + ttl seconds.

        Args:
            key:   Cache key string.
            value: Arbitrary serializable value to cache.
            ttl:   Seconds until this entry expires and is evicted.
        """
        with self._lock:
            self._store[key] = {
                "value":      value,
                "expires_at": time.time() + ttl,
            }
        logger.debug("Cache SET key='%s' ttl=%ds", key, ttl)

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached value if it exists and has not expired.

        Args:
            key: Cache key to look up.

        Returns:
            The stored value, or None if the key is missing or stale.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.time() > entry["expires_at"]:
                del self._store[key]
                logger.debug("Cache MISS (expired) key='%s'", key)
                return None
            logger.debug("Cache HIT key='%s'", key)
            return entry["value"]

    def invalidate(self, key: str) -> None:
        """
        Manually evict a single cache entry.

        Args:
            key: The key to remove.
        """
        with self._lock:
            self._store.pop(key, None)
        logger.debug("Cache INVALIDATE key='%s'", key)

    def clear_expired(self) -> int:
        """
        Purge all entries whose TTL has elapsed.
        Call periodically (e.g. on a background task) to prevent memory growth.

        Returns:
            Number of entries removed.
        """
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._store.items() if v["expires_at"] <= now]
            for k in expired:
                del self._store[k]
        if expired:
            logger.debug("Cache GC — evicted %d expired entries", len(expired))
        return len(expired)

    @property
    def size(self) -> int:
        """Returns the current number of entries (including potentially stale ones)."""
        with self._lock:
            return len(self._store)


# ── Singleton ──────────────────────────────────────────────────────────────────
# Import this instance everywhere — do NOT instantiate TTLCache directly.
cache = TTLCache()


def init_cache() -> None:
    """
    Called once at application startup to log cache initialisation.
    The cache itself is ready at import time; this is a hook for future
    warm-up logic (e.g. pre-loading frequent sessions).
    """
    logger.info("TTL cache initialized — entries=%d", cache.size)
