"""OddsPortal export (user-supplied scrape) -> clean match-winner quotes.

The file stays local in data/raw/oddsportal/ (gitignored): it was scraped from OddsPortal,
whose terms restrict scraping and redistribution, so nothing row-level from it is committed.

A row is kept only if it is "enriched" (per-bookmaker closing odds, a match id) and its odds
belong to that match. Dropped:
  junk               header text parsed as a match ("vs", "Odds, Predictions and H2H Results");
  not_enriched       no per-bookmaker average odds;
  stale_shared_link  one match link and one price attached to meetings on different dates
                     (an older scraper bug: a rematch got another match's odds);
  duplicate_match_id the same match id on more than one row.
Prices are OddsPortal's average closing odds over the listed bookmakers (median 6, mostly
soft and crypto books; no Pinnacle). The margin is removed proportionally.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shyarly import paths
from shyarly.data.market import fair_probs

ODDSPORTAL_FILE = paths.RAW_DIR / "oddsportal" / "lol_historical_odds_2022_2026.csv"
JUNK_NAMES = {"vs", "Odds, Predictions and H2H Results"}


def load_oddsportal(path: Path = ODDSPORTAL_FILE) -> tuple[pd.DataFrame, dict[str, int]]:
    d = pd.read_csv(path, low_memory=False)
    dropped: dict[str, int] = {}

    def drop(mask, reason):
        nonlocal d
        n = int(mask.sum())
        if n:
            dropped[reason] = n
        d = d[~mask]

    drop(d["home_team"].isin(JUNK_NAMES) | d["away_team"].isin(JUNK_NAMES), "junk")
    drop(d["odds_average_home"].isna() | d["odds_average_away"].isna(), "not_enriched")
    url = d["match_url"].fillna("")
    shared = d[url.str.contains("#") & d.duplicated("match_url", keep=False)].groupby("match_url").agg(
        dates=("match_date", "nunique"), prices=("odds_average_home", "nunique"))
    stale = set(shared[(shared["dates"] > 1) & (shared["prices"] == 1)].index)
    drop(d["match_url"].isin(stale), "stale_shared_link")
    drop(d["match_id"].notna() & d.duplicated("match_id", keep=False), "duplicate_match_id")

    start = pd.to_datetime(d["match_date"], format="%d %b %Y", errors="coerce").dt.tz_localize("UTC")
    drop(start.isna(), "bad_date")
    start = start[start.notna()]
    fair = [fair_probs(h, a) for h, a in zip(d["odds_average_home"], d["odds_average_away"])]
    opening = [fair_probs(h, a)[0] if pd.notna(h) and pd.notna(a) else float("nan")
               for h, a in zip(d["odds_opening_home"], d["odds_opening_away"])]
    q = pd.DataFrame({
        "fixture_id": d["match_id"].fillna(d["match_url"]).astype(str).to_numpy(),
        "tournament": d["tournament"].to_numpy(), "season": d["season"].astype(int).to_numpy(),
        "start": start.to_numpy(), "team_1": d["home_team"].to_numpy(), "team_2": d["away_team"].to_numpy(),
        "p_mkt_series": [f[0] for f in fair], "overround": [f[2] for f in fair], "p_open_series": opening,
        "n_books": d["bookmaker_count"].to_numpy(),
    })
    q["start"] = pd.to_datetime(q["start"], utc=True)
    return q.reset_index(drop=True), dropped
