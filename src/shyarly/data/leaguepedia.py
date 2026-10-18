"""Throttled, cached client for the Leaguepedia Cargo API.

Every response is written to data/raw/leaguepedia/<sha1(params)>.json and served
from there on later calls, so a query hits the network at most once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import requests

from shyarly import paths
from shyarly.config import LeaguepediaSource

log = logging.getLogger(__name__)


class CargoClient:
    def __init__(self, cfg: LeaguepediaSource, cache_dir: Path = paths.LEAGUEPEDIA_RAW_DIR):
        self.cfg = cfg
        self.cache_dir = cache_dir
        self._last_call = 0.0
        self._session = requests.Session()
        self._session.headers["User-Agent"] = cfg.user_agent

    def _cache_path(self, params: dict) -> Path:
        key = json.dumps(params, sort_keys=True)
        return self.cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()}.json"

    def _throttle(self) -> None:
        wait = self.cfg.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def query(self, tables: str, fields: str, where: str = "", limit: int = 500, offset: int = 0,
              **extra: str) -> list[dict]:
        params = {
            "action": "cargoquery", "format": "json", "tables": tables, "fields": fields,
            "where": where, "limit": str(limit), "offset": str(offset), **extra,
        }
        path = self._cache_path(params)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["cargoquery"]

        for attempt in range(self.cfg.max_retries):
            self._throttle()
            r = self._session.get(self.cfg.api_url, params=params, timeout=60)
            r.raise_for_status()
            body = r.json()
            if body.get("error", {}).get("code") == "ratelimited":
                delay = self.cfg.backoff_s * (attempt + 1)
                log.warning("Leaguepedia rate-limited; sleeping %.0fs", delay)
                time.sleep(delay)
                continue
            if "error" in body:
                raise RuntimeError(f"Cargo error: {body['error']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"params": params, **body}), encoding="utf-8")
            return body["cargoquery"]
        raise RuntimeError("Leaguepedia still rate-limited after retries")
