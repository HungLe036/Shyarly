"""Compact market-odds table: data/manual/market_odds.csv (committed).

The raw OddsPapi responses stay in data/raw/odds/ (local, gitignored). This table keeps,
per match and bookmaker, only the closing match-winner quote: the last price for each side
before the scheduled start, and the fair probabilities after removing the bookmaker margin
(proportional method). Rows are append-only, like the prediction log: a match's first
recorded closing quote is never overwritten.

Note: OddsPapi's terms may restrict redistributing its data; keep this file out of any
public copy of the repo unless the terms allow it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from shyarly import paths

MARKET_ODDS = paths.MANUAL_DIR / "market_odds.csv"
MARKET_COLUMNS = ["fixture_id", "tournament", "start_time_utc", "team_1", "team_2", "bookmaker", "market",
                  "price_1", "price_2", "p1_fair", "p2_fair", "overround", "quote_time_utc", "n_quotes",
                  "source", "fetched_at_utc"]


def fair_probs(price_1: float, price_2: float) -> tuple[float, float, float]:
    """(p1, p2, overround) from decimal odds; the margin is removed proportionally."""
    i1, i2 = 1.0 / price_1, 1.0 / price_2
    return i1 / (i1 + i2), i2 / (i1 + i2), i1 + i2 - 1.0


def _snapshots(body: dict, bookmaker: str, market: str, outcome: str) -> list[dict]:
    try:
        players = body["bookmakers"][bookmaker]["markets"][market]["outcomes"][outcome]["players"]
    except (KeyError, TypeError):
        return []
    snaps = players.get("0") if isinstance(players, dict) else None
    return snaps if isinstance(snaps, list) else []


def closing_quote(body: dict, start: pd.Timestamp, bookmaker: str, market: str = "171") -> dict[str, Any] | None:
    """Last pre-start price for both sides of a two-way market, or None if either side has none."""
    start = pd.Timestamp(start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    sides, n = [], 0
    for outcome in (market, str(int(market) + 1)):
        snaps = _snapshots(body, bookmaker, market, outcome)
        n += len(snaps)
        pre = [(pd.Timestamp(s["createdAt"]), float(s["price"])) for s in snaps
               if s.get("price") and pd.Timestamp(s["createdAt"]) < start]
        if not pre:
            return None
        sides.append(max(pre))
    (t1, p1), (t2, p2) = sides
    return {"price_1": p1, "price_2": p2, "quote_time_utc": max(t1, t2).tz_convert("UTC").isoformat(), "n_quotes": n}


def append_market_rows(path: Path, rows: list[dict]) -> int:
    """Append rows whose (fixture_id, bookmaker, market) is not in the table yet."""
    path = Path(path)
    old = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=MARKET_COLUMNS)
    have = set(zip(old["fixture_id"], old["bookmaker"], old["market"].astype(str)))
    new = [r for r in rows if (str(r["fixture_id"]), r["bookmaker"], str(r["market"])) not in have]
    if new or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        out = pd.concat([old, pd.DataFrame(new, columns=MARKET_COLUMNS)], ignore_index=True)
        out[MARKET_COLUMNS].to_csv(path, index=False)
    return len(new)
