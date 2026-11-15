"""
v2/simulation/cache.py
----------------------
Versioned pickle cache. Each entry stores the key it was built under (seasons,
feature lists, model settings, a schema version); a load with a different key is a
miss, so a feature or setting change can never be scored against stale rows.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path


def cache_key(**parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def load(path, key: str):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            entry = pickle.load(fh)
    except Exception:
        return None
    if not isinstance(entry, dict) or entry.get("key") != key:
        return None
    return entry["data"]


def save(path, key: str, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump({"key": key, "data": data}, fh)
