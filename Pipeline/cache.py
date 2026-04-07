"""
DDDS - API Response Cache
==========================
Provides deterministic read/write caching for all LLM API calls.

Every call is keyed by a SHA-256 hash of its full input payload
(model + system prompt + user content). Responses are stored as
individual JSON files in data/cache/.

Usage
-----
    from cache import get_cached, save_to_cache

    payload = {"model": "gpt-4o-mini", "system": SYSTEM_PROMPT, "user": user_prompt}

    if use_cached:
        cached = get_cached(payload)
        if cached:
            return cached

    # ... make live API call ...

    save_to_cache(payload, response)

Reproducing outputs
-------------------
With a populated cache (data/cache/), pass --use-cached to any pipeline
script to replay all API responses at zero API cost:

    python "run 8 - graph RAG.py" --all --use-cached
    python "run 1 - Topic Extraction.py" --input data/passages.csv --mode sync --use-cached
"""

import hashlib
import json
import os
from pathlib import Path

# Cache directory is always data/cache/ relative to the project root
# (one level above this file, which lives in Pipeline/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "cache")


def _cache_key(payload: dict) -> str:
    """Returns a deterministic SHA-256 hex digest for the given payload dict."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_cached(payload: dict, cache_dir: str = None) -> dict | None:
    """
    Returns the cached response for this payload, or None if not cached.

    Args:
        payload:   Dict describing the full API request (model, system, user content).
        cache_dir: Override the default cache directory.
    """
    dir_ = cache_dir or DEFAULT_CACHE_DIR
    path = Path(dir_) / f"{_cache_key(payload)}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_to_cache(payload: dict, response: dict, cache_dir: str = None) -> None:
    """
    Saves an API response to the cache.

    Args:
        payload:   Dict describing the full API request (must match what was passed to get_cached).
        response:  The parsed API response (dict) to store.
        cache_dir: Override the default cache directory.
    """
    dir_ = cache_dir or DEFAULT_CACHE_DIR
    os.makedirs(dir_, exist_ok=True)
    path = Path(dir_) / f"{_cache_key(payload)}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2, ensure_ascii=False)


def cache_stats(cache_dir: str = None) -> dict:
    """Returns count and total size of cached entries."""
    dir_ = cache_dir or DEFAULT_CACHE_DIR
    files = list(Path(dir_).glob("*.json")) if Path(dir_).exists() else []
    total_bytes = sum(f.stat().st_size for f in files)
    return {
        "entries": len(files),
        "size_mb": round(total_bytes / 1024 / 1024, 2),
        "cache_dir": dir_,
    }
