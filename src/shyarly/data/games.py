"""Build the one-row-per-game table (data/processed/games.parquet) from Oracle's Elixir.

Point-in-time rule: every derived column of a game uses only that game and games
strictly earlier in time, so appending future rows never changes past rows. In
particular a team's region at an international event is its home region as of
the event, taken from the domestic games it had already played.

OE `date` is UTC: the Worlds 2026 Play-In's first game is stamped 18:16, and the
lolesports primer lists the start as 11 AM PDT (18:00 UTC).
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from shyarly import paths
from shyarly.config import LeagueMap

log = logging.getLogger(__name__)

OE_COLUMNS = ["gameid", "date", "league", "game", "patch", "playoffs", "split", "position", "side", "teamname", "result",
              "golddiffat15"]

GAMES_SCHEMA: dict[str, str] = {
    "game_id": "str",
    "date_utc": "datetime64[us, UTC]",
    "year": "int16",
    "patch": "str",
    "league": "str",
    "event": "str",
    "stage": "str",                 # regular | playoffs | qualifier | main
    "series_id": "str",
    "game_in_series": "int8",
    "blue_team": "str",
    "red_team": "str",
    "blue_team_season_id": "str",
    "red_team_season_id": "str",
    "blue_region": "str",           # NaN when the team has no known home league
    "red_region": "str",
    "winner": "str",                # blue | red
    "blue_win": "int8",
    "gd15": "float64",              # blue minus red gold at 15:00; NaN when OE has no timeline data
    "cross_region": "bool",         # both regions known and different
    "scored": "bool",               # main-event game of a backtest-scored international event
    "data_source": "str",
}

SERIES_GAP = pd.Timedelta(hours=12)
DUPLICATE_WINDOW = pd.Timedelta(minutes=5)


def slugify(name: str) -> str:
    """'Gen.G' -> 'gen-g'. Names with non-ASCII characters get a short hash suffix so
    that, e.g., 'LØS' and 'LS' or two all-CJK names never collide."""
    ascii_ = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_.lower()).strip("-")
    if not name.isascii():
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:6]
        slug = f"{slug}-{digest}" if slug else digest
    return slug


def read_oe(oe_dir: Path = paths.OE_RAW_DIR, years: range = range(2022, 2027)) -> pd.DataFrame:
    """Team-level rows (two per game) from the cached OE yearly CSVs."""
    frames = []
    for y in years:
        path = oe_dir / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv"
        df = pd.read_csv(path, usecols=OE_COLUMNS, dtype={"patch": "str", "split": "str"}, low_memory=False)
        frames.append(df[df["position"] == "team"].drop(columns="position"))
    return pd.concat(frames, ignore_index=True)


def _to_wide(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["date_utc"] = pd.to_datetime(rows["date"]).dt.tz_localize("UTC").astype("datetime64[us, UTC]")
    ok = rows.groupby("gameid").agg(n=("side", "size"), sides=("side", "nunique"), wins=("result", "sum"))
    bad = ok.index[(ok["n"] != 2) | (ok["sides"] != 2) | (ok["wins"] != 1)]
    if len(bad):
        log.warning("Dropping %d games without exactly one blue, one red and one winner", len(bad))
    rows = rows[~rows["gameid"].isin(bad)]

    blue = rows[rows["side"] == "Blue"].set_index("gameid")
    red = rows[rows["side"] == "Red"].set_index("gameid")
    g = blue[["date_utc", "league", "game", "patch", "playoffs", "split"]].copy()
    g["blue_team"] = blue["teamname"]
    g["red_team"] = red.loc[g.index, "teamname"]
    g["blue_win"] = blue["result"].astype("int8")
    g["gd15"] = pd.to_numeric(blue["golddiffat15"], errors="coerce") if "golddiffat15" in blue else np.nan
    g = g.reset_index().rename(columns={"gameid": "game_id"})
    g = g.sort_values(["date_utc", "game_id"], kind="stable").reset_index(drop=True)
    return _drop_double_records(g)


def _drop_double_records(g: pd.DataFrame) -> pd.DataFrame:
    """OE occasionally lists one game twice under two gameids (seen in DDH 2023 and
    UPL 2022). Keep the first record when the same league and blue/red teams start
    within DUPLICATE_WINDOW of each other."""
    key = ["league", "blue_team", "red_team"]
    prev = g.groupby(key)["date_utc"].shift()
    dup = (g["date_utc"] - prev) <= DUPLICATE_WINDOW
    if dup.any():
        log.warning("Dropping %d double-recorded games: %s", int(dup.sum()), g.loc[dup, "game_id"].tolist())
    return g[~dup].reset_index(drop=True)


def _home_region(counts: dict[tuple[int, str], int]) -> str | None:
    """Most-played tier-1 region, else most-played tier-2 region. Ties break alphabetically."""
    for tier in (1, 2):
        c = {r: n for (t, r), n in counts.items() if t == tier}
        if c:
            return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return None


def _assign_regions(g: pd.DataFrame, leagues: LeagueMap) -> tuple[list, list]:
    """Region per side per game. A game in a mapped league takes that league's region;
    any other game takes the team's home region from strictly earlier games."""
    year = g["date_utc"].dt.year
    league_region = [leagues.lookup(lg, int(y)) for lg, y in zip(g["league"], year)]

    long = pd.concat([
        pd.DataFrame({"i": g.index, "side": "blue", "team": g["blue_team"]}),
        pd.DataFrame({"i": g.index, "side": "red", "team": g["red_team"]}),
    ])
    long["t"] = g.loc[long["i"], "date_utc"].to_numpy()
    long = long.sort_values(["team", "t", "i"], kind="stable")

    out = {"blue": [None] * len(g), "red": [None] * len(g)}
    for team, rows in long.groupby("team", sort=False):
        by_year: dict[int, Counter] = defaultdict(Counter)
        pending: list[tuple[int, int, str]] = []   # updates applied once time moves past this timestamp
        last_t = None
        for i, side, t in zip(rows["i"], rows["side"], rows["t"]):
            if t != last_t:
                for y, tier, region in pending:
                    by_year[y][(tier, region)] += 1
                pending, last_t = [], t
            y = t.year
            lr = league_region[i]
            if lr is not None:
                region, tier = lr
                out[side][i] = region
                pending.append((y, tier, region))
            else:
                home = _home_region(by_year[y]) if by_year.get(y) else None
                if home is None:
                    prior = [yy for yy in by_year if yy < y and by_year[yy]]
                    home = _home_region(by_year[max(prior)]) if prior else None
                out[side][i] = home
    return out["blue"], out["red"]


def _assign_series(g: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    a = g[["blue_team", "red_team"]].min(axis=1)
    b = g[["blue_team", "red_team"]].max(axis=1)
    key = g["league"] + "|" + a + "|" + b
    order = g.assign(_key=key).sort_values(["_key", "date_utc", "game"], kind="stable")
    prev_t = order.groupby("_key")["date_utc"].shift()
    prev_game = order.groupby("_key")["game"].shift()
    new = (
        prev_t.isna()
        | (order["date_utc"] - prev_t > SERIES_GAP)
        | (order["game"] <= 1)
        | (order["game"] != prev_game + 1)
    )
    seq = new.cumsum()
    first_t = order.groupby(seq)["date_utc"].transform("min")
    sid = (
        order["league"] + "|" + first_t.dt.strftime("%Y%m%dT%H%M") + "|"
        + a.loc[order.index].map(slugify) + "|" + b.loc[order.index].map(slugify)
    )
    gis = order.groupby(seq).cumcount() + 1
    return sid.reindex(g.index), gis.reindex(g.index)


def build_games(rows: pd.DataFrame, leagues: LeagueMap) -> pd.DataFrame:
    """Team-level OE rows -> one row per game with the GAMES_SCHEMA columns."""
    g = _to_wide(rows)
    year = g["date_utc"].dt.year

    blue_region, red_region = _assign_regions(g, leagues)
    g["blue_region"], g["red_region"] = blue_region, red_region
    g["cross_region"] = g["blue_region"].notna() & g["red_region"].notna() & (g["blue_region"] != g["red_region"])

    is_intl = g["league"].isin(leagues.international_events)
    # An international event's main phase starts at its first cross-region game; same-region
    # games before that are regional qualifiers that OE files under the event's league code.
    first_cross = g[is_intl & g["cross_region"]].groupby([g["league"], year])["date_utc"].min()
    start = pd.Series(list(zip(g["league"], year))).map(first_cross).set_axis(g.index)
    intl_stage = (g["date_utc"] >= start).map({True: "main", False: "qualifier"})
    dom_stage = (pd.to_numeric(g["playoffs"], errors="coerce") == 1).map({True: "playoffs", False: "regular"})
    g["stage"] = intl_stage.where(is_intl, dom_stage)
    g["scored"] = g["league"].isin(leagues.scored_events) & (g["stage"] == "main")

    split = g["split"].fillna("").astype("str")
    base = g["league"] + " " + year.astype("str")
    g["event"] = base.where(is_intl | (split == ""), base + " " + split)

    g["series_id"], g["game_in_series"] = _assign_series(g)
    g["year"] = year
    g["blue_team_season_id"] = g["blue_team"].map(slugify) + "_" + year.astype("str")
    g["red_team_season_id"] = g["red_team"].map(slugify) + "_" + year.astype("str")
    g["winner"] = g["blue_win"].map({1: "blue", 0: "red"})
    g["data_source"] = "oe"

    out = g[list(GAMES_SCHEMA)]
    return out.astype(GAMES_SCHEMA)


def build_series(games: pd.DataFrame) -> pd.DataFrame:
    """One row per series, derived from games. `inferred_best_of` = 2 * winner's wins - 1."""
    g = games.copy()
    g["winner_team"] = g["blue_team"].where(g["winner"] == "blue", g["red_team"])
    g["team_a"] = g[["blue_team", "red_team"]].min(axis=1)
    g["team_b"] = g[["blue_team", "red_team"]].max(axis=1)
    g["a_win"] = (g["winner_team"] == g["team_a"]).astype(int)
    s = g.groupby("series_id").agg(
        date_utc=("date_utc", "min"), league=("league", "first"), event=("event", "first"),
        stage=("stage", "first"), scored=("scored", "first"), team_a=("team_a", "first"),
        team_b=("team_b", "first"), a_region=("blue_region", "first"), n_games=("game_id", "size"),
        wins_a=("a_win", "sum"),
    ).reset_index()
    s["wins_b"] = s["n_games"] - s["wins_a"]
    s["winner_team"] = s["team_a"].where(s["wins_a"] > s["wins_b"], s["team_b"]).where(s["wins_a"] != s["wins_b"])
    s["inferred_best_of"] = (2 * s[["wins_a", "wins_b"]].max(axis=1) - 1).astype("int8")
    return s.drop(columns="a_region").sort_values("date_utc").reset_index(drop=True)


def write_processed(leagues: LeagueMap, out_dir: Path = paths.PROCESSED_DIR) -> pd.DataFrame:
    games = build_games(read_oe(), leagues)
    out_dir.mkdir(parents=True, exist_ok=True)
    games.to_parquet(out_dir / paths.GAMES_PARQUET.name, index=False)
    build_series(games).to_parquet(out_dir / paths.SERIES_PARQUET.name, index=False)
    return games
