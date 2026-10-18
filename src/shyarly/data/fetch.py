"""Download and cache raw source files under data/raw/.

Each cached file is recorded in a manifest.json next to it, with its URL, size,
sha256 and fetch time, so we can tell exactly which snapshot a run used.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import requests

from shyarly import paths
from shyarly.config import Sources

log = logging.getLogger(__name__)

CHUNK = 1 << 20
TIMEOUT_S = 120


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Stream to a temp file and rename, so a partial download never looks cached."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=TIMEOUT_S) as r:
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "text/html" in ctype:
            raise RuntimeError(f"Expected a file but got HTML from {url} (Drive interstitial?)")
        with tmp.open("wb") as f:
            for block in r.iter_content(CHUNK):
                f.write(block)
    tmp.replace(dest)


def _load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _record(manifest: dict, key: str, url: str, dest: Path) -> None:
    manifest[key] = {
        "url": url,
        "path": dest.relative_to(paths.ROOT).as_posix(),
        "bytes": dest.stat().st_size,
        "sha256": _sha256(dest),
        "fetched_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def fetch_oracles_elixir(
    sources: Sources,
    years: list[int] | None = None,
    refresh: bool = False,
    out_dir: Path = paths.OE_RAW_DIR,
) -> dict:
    """Cache OE yearly CSVs. Existing files are kept unless `refresh` is set and
    the year is listed in `refresh_years` (OE only updates the current year)."""
    oe = sources.oracles_elixir
    years = years or sorted(oe.file_ids)
    manifest_path = out_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)

    for year in years:
        dest = out_dir / oe.local_name(year)
        stale = refresh and year in oe.refresh_years
        if dest.exists() and not stale:
            log.info("OE %s: cached (%s)", year, dest.name)
            if str(year) not in manifest:
                _record(manifest, str(year), oe.url(year), dest)
            continue
        log.info("OE %s: downloading", year)
        _download(oe.url(year), dest)
        _record(manifest, str(year), oe.url(year), dest)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def fetch_rules(sources: Sources, out_dir: Path = paths.RULES_RAW_DIR) -> dict:
    """Cache the official ruleset PDF(s) listed under `rules:` in sources.yaml."""
    manifest_path = out_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    for key, url in sources.rules.items():
        dest = out_dir / f"{key}.pdf"
        if not dest.exists():
            _download(url, dest)
        _record(manifest, key, url, dest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
