"""OddsPapi client (betting odds), built around a tight free quota.

* Every response is cached under data/raw/odds/ by endpoint + parameters (the API key is
  never part of the cache key, a log line or an error message). A cached call is never repeated.
* Billable calls (every endpoint except /historical-odds and /account) are recorded in
  data/raw/odds/ledger.jsonl. A billable call is refused once this month's ledger reaches
  `monthly_budget` (config/sources.yaml `oddspapi`).
* The key is read from ODDSPAPI_KEY in the environment, else from a gitignored .env file.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from shyarly import paths
from shyarly.config import _read_yaml

FREE_ENDPOINTS = {"historical-odds", "account"}
CACHE_DIR = paths.RAW_DIR / "odds"
ENV_FILE = paths.ROOT / ".env"


class OddsPapiError(RuntimeError):
    pass


class BudgetExceeded(OddsPapiError):
    pass


def load_key(env_file: Path = ENV_FILE) -> str:
    key = os.environ.get("ODDSPAPI_KEY", "").strip()
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "ODDSPAPI_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        raise OddsPapiError("No OddsPapi key: set ODDSPAPI_KEY or put ODDSPAPI_KEY=... in .env (gitignored)")
    return key


def _transport(url: str, params: dict[str, Any]) -> tuple[int, Any]:
    r = requests.get(url, params=params, timeout=60)
    try:
        body = r.json()
    except ValueError:
        body = r.text[:500]
    return r.status_code, body


class OddsPapi:
    def __init__(self, key: str | None = None, cache_dir: Path = CACHE_DIR, cfg: dict | None = None,
                 transport: Callable[[str, dict], tuple[int, Any]] = _transport,
                 now: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self.cfg = cfg or _read_yaml(paths.SOURCES_CONFIG)["oddspapi"]
        self._key = key
        self.cache_dir = cache_dir
        self.transport = transport
        self.now = now
        self._last = 0.0

    # ------------------------------------------------------------ budget
    @property
    def ledger(self) -> Path:
        return self.cache_dir / "ledger.jsonl"

    def billable_this_month(self) -> int:
        if not self.ledger.exists():
            return 0
        month = self.now().strftime("%Y-%m")
        return sum(1 for line in self.ledger.read_text(encoding="utf-8").splitlines()
                   if line and json.loads(line)["at"].startswith(month))

    def remaining(self) -> int:
        return self.cfg["monthly_budget"] - self.billable_this_month()

    # ------------------------------------------------------------- calls
    def _cache_path(self, endpoint: str, params: dict[str, Any]) -> Path:
        key = json.dumps({"endpoint": endpoint, **{k: str(v) for k, v in sorted(params.items())}}, sort_keys=True)
        return self.cache_dir / endpoint / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"

    def get(self, endpoint: str, refresh: bool = False, **params) -> Any:
        """GET /v4/<endpoint>. Cached unless `refresh`; billable calls are budget-checked."""
        path = self._cache_path(endpoint, params)
        if path.exists() and not refresh:
            return json.loads(path.read_text(encoding="utf-8"))["body"]
        billable = endpoint not in FREE_ENDPOINTS
        if billable and self.remaining() <= 0:
            raise BudgetExceeded(f"OddsPapi monthly budget of {self.cfg['monthly_budget']} billable calls is used up")
        wait = self.cfg.get("min_interval_s", 1.0) - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        key = self._key or load_key()
        status, body = self.transport(f"{self.cfg['base_url']}/{endpoint}", {**params, "apiKey": key})
        self._last = time.monotonic()
        at = self.now().isoformat()
        if billable:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"at": at, "endpoint": endpoint, "params": params, "status": status}) + "\n")
        if status != 200:
            text = json.dumps(body)[:300] if not isinstance(body, str) else body[:300]
            raise OddsPapiError(f"{endpoint} returned HTTP {status}: {text.replace(key, '***')}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at": at, "endpoint": endpoint, "params": params, "body": body}),
                        encoding="utf-8")
        return body
