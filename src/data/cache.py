"""File-based TTL cache. Keys are hashed; metadata stored alongside data."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional


class Cache:
    def __init__(self, cache_dir: str = "cache"):
        self._root = Path(cache_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: str) -> tuple[Path, Path]:
        h = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self._root / f"{h}.json", self._root / f"{h}.meta.json"

    def get(self, key: str, ttl_hours: float) -> Optional[Any]:
        data_path, meta_path = self._paths(key)
        if not data_path.exists() or not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        age_hours = (time.time() - meta["saved_at"]) / 3600
        if age_hours > ttl_hours:
            return None
        return json.loads(data_path.read_text())

    def set(self, key: str, value: Any) -> None:
        data_path, meta_path = self._paths(key)
        data_path.write_text(json.dumps(value, default=str))
        meta_path.write_text(json.dumps({"saved_at": time.time(), "key": key}))

    def invalidate(self, key: str) -> None:
        for p in self._paths(key):
            if p.exists():
                p.unlink()
