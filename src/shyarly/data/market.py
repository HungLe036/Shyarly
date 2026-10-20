"""Compact market-odds table: data/manual/market_odds.csv (local only, gitignored: OddsPapi terms restrict redistribution).

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
from shyarly.data.oddspapi import OddsPapiError

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


def closing_quote(body: dict, start: pd.Timestamp, bookmaker: str, market: str = "181") -> dict[str, Any] | None:
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


def tournament_rows(client, tournament_id: int, bookmaker: str, markets: list[int],
                    refresh_fixtures: bool = False) -> tuple[list[dict], int]:
    """Closing quotes for every finished fixture of a tournament. The fixture list is one
    billable call (cached); the odds histories are free. Returns (rows, fixtures without odds)."""
    fixtures = client.get("fixtures", refresh=refresh_fixtures, tournamentId=tournament_id)
    rows, missing = [], 0
    for f in fixtures:
        start = pd.Timestamp(f["startTime"])
        if f.get("statusId") != 2:                    # only finished matches: their history is final
            continue
        try:
            hist = client.get("historical-odds", fixtureId=f["fixtureId"], bookmakers=bookmaker)
        except OddsPapiError as e:
            if e.status != 404:                       # 404 = "No historical odds found" for this fixture
                raise
            missing += 1
            continue
        got = False
        for m in markets:
            q = closing_quote(hist, start, bookmaker, str(m))
            if q is None:
                continue
            got = True
            p1, p2, over = fair_probs(q["price_1"], q["price_2"])
            rows.append({
                "fixture_id": f["fixtureId"], "tournament": f.get("tournamentName"), "start_time_utc": start.isoformat(),
                "team_1": f.get("participant1Name"), "team_2": f.get("participant2Name"), "bookmaker": bookmaker,
                "market": m, "price_1": q["price_1"], "price_2": q["price_2"], "p1_fair": round(p1, 6),
                "p2_fair": round(p2, 6), "overround": round(over, 6), "quote_time_utc": q["quote_time_utc"],
                "n_quotes": q["n_quotes"], "source": "oddspapi /v4/historical-odds",
                "fetched_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            })
        missing += not got
    return rows, missing


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
