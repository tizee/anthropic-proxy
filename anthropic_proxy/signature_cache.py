"""In-memory signature cache for Antigravity Claude thinking blocks."""

from __future__ import annotations

import time


# In-memory cache: key -> {"signature": str, "timestamp": float}
_SIGNATURE_CACHE: dict[str, dict[str, float | str]] = {}

# TTL for cache entries (30 minutes)
_CACHE_TTL_SECONDS = 30 * 60

# Maximum cache size
_MAX_CACHE_SIZE = 1000


def _make_key(session_id: str, text: str) -> str:
    text_key = text[:200]
    return f"{session_id}:{text_key}"


def _cleanup_expired() -> None:
    now = time.time()
    expired = [
        key for key, entry in _SIGNATURE_CACHE.items()
        if now - float(entry["timestamp"]) > _CACHE_TTL_SECONDS
    ]
    for key in expired:
        _SIGNATURE_CACHE.pop(key, None)


def cache_signature(session_id: str, thinking_text: str, signature: str) -> None:
    if not session_id or not thinking_text or not signature:
        return

    if len(_SIGNATURE_CACHE) >= _MAX_CACHE_SIZE:
        _cleanup_expired()

    _SIGNATURE_CACHE[_make_key(session_id, thinking_text)] = {
        "signature": signature,
        "timestamp": time.time(),
    }


def get_cached_signature(session_id: str, thinking_text: str) -> str | None:
    if not session_id or not thinking_text:
        return None

    key = _make_key(session_id, thinking_text)
    entry = _SIGNATURE_CACHE.get(key)
    if not entry:
        return None

    if time.time() - float(entry["timestamp"]) > _CACHE_TTL_SECONDS:
        _SIGNATURE_CACHE.pop(key, None)
        return None

    signature = entry.get("signature")
    return str(signature) if signature else None


def clear_session_cache(session_id: str) -> None:
    if not session_id:
        return
    prefix = f"{session_id}:"
    for key in list(_SIGNATURE_CACHE.keys()):
        if key.startswith(prefix):
            _SIGNATURE_CACHE.pop(key, None)


def clear_all_cache() -> None:
    _SIGNATURE_CACHE.clear()
