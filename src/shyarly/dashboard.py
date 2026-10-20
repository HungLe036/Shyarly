"""Read-only loaders for the Streamlit page (`shyarly.app`). Pure pandas, no Streamlit import,
so they are testable without the `app` dependency group. Every loader returns an empty frame
(or None) when its file does not exist yet, e.g. before the first `shyarly update`."""

from pathlib import Path

import pandas as pd

from shyarly import paths

RECORDS = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]
STAGES = {"qualify": "Quarterfinals", "semis": "Semifinals", "final": "Final", "champion": "Champion"}

# Region identity, in a fixed order (colours never follow rank). Hues: the validated categorical
# order of the dataviz reference palette, slots 1-6, dark-mode steps (the page is dark; all six
# pass colour-blind separation and >= 3:1 contrast on a dark surface).
REGIONS = {
    "LCK": ("Korea", "#3987e5"),
    "LPL": ("China", "#d95926"),
    "LEC": ("Europe", "#199e70"),
    "LCS": ("North America", "#c98500"),
    "LCP": ("Asia-Pacific", "#d55181"),
    "CBLOL": ("Brazil", "#008300"),
}


def _read_csv(path: Path, **kw) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, **kw)


def title_odds(path: Path = paths.TOURNAMENT_ODDS_CSV) -> pd.DataFrame:
    """Quarterfinal (= Swiss qualify), semifinal, final and champion % per team, best first."""
    df = _read_csv(path)
    if df.empty:
        return df
    cols = [c for c in STAGES if c in df.columns]
    out = df[["team", "name", "region", "league_seed", *cols]].rename(columns=STAGES)
    return out.sort_values(STAGES[cols[-1]], ascending=False, ignore_index=True)


def swiss_records(path: Path = paths.SWISS_ODDS_CSV) -> pd.DataFrame:
    """Probability of each final Swiss record (3-0 … 0-3) per team, by qualify %."""
    df = _read_csv(path)
    if df.empty:
        return df
    out = df[["team", "name", "region", "qualify", *RECORDS]]
    return out.sort_values("qualify", ascending=False, ignore_index=True)


def odds_history(daily_dir: Path = paths.DAILY_DIR) -> pd.DataFrame:
    """All daily snapshots stacked in long form: as_of, team, stage, p."""
    files = sorted(daily_dir.glob("*_odds.csv")) if daily_dir.exists() else []
    frames = [pd.read_csv(f) for f in files if f.stat().st_size > 0]
    if not frames:
        return pd.DataFrame(columns=["as_of", "team", "stage", "p"])
    df = pd.concat(frames, ignore_index=True)
    df["as_of"] = pd.to_datetime(df["as_of"])
    cols = [c for c in STAGES if c in df.columns]
    long = df.melt(id_vars=["as_of", "team"], value_vars=cols, var_name="stage", value_name="p")
    long["stage"] = long["stage"].map(STAGES)
    return long.sort_values(["team", "stage", "as_of"], ignore_index=True)


def _match_id(r) -> str:
    a, b = sorted((r["team_a"], r["team_b"]))
    return f"{r['stage']}-{r['round']}-{a}-{b}"


def predictions_vs_results(log_path: Path = paths.PREDICTIONS_LOG,
                           results_path: Path = paths.MANUAL_RESULTS) -> pd.DataFrame:
    """First logged prediction per (match, model, mode) next to the entered result, if any."""
    log = _read_csv(log_path, dtype={"round": str})
    if log.empty:
        return log
    log = log.assign(model=log["model"].fillna("bt")) if "model" in log else log.assign(model="bt")
    first = log.drop_duplicates(["match_id", "model", "mode"], keep="first")
    res = _read_csv(results_path, dtype={"round": str})
    if res.empty:
        first = first.assign(winner=pd.NA, score=pd.NA)
    else:
        res = res.assign(match_id=res.apply(_match_id, axis=1))[["match_id", "winner", "score"]]
        first = first.merge(res, on="match_id", how="left")
    keep = ["match_id", "match_date_utc", "stage", "round", "team_a", "team_b", "best_of", "model", "mode",
            "p_series_a", "p_game_a", "winner", "score", "logged_at_utc"]
    return first[keep].sort_values(["match_date_utc", "model", "mode"], ignore_index=True)


def match_forecasts(path: Path = paths.MATCH_FORECASTS) -> pd.DataFrame:
    """Current forecast for drawn, unplayed matches (written by `shyarly forecast` / `update`)."""
    df = _read_csv(path, dtype={"round": str})
    return df.assign(winner=pd.NA, score=pd.NA) if not df.empty else df


def with_forecasts(preds: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Logged predictions plus the current forecast for matches with no logged live prediction yet."""
    if forecasts.empty:
        return preds
    if preds.empty:
        return forecasts
    live = preds.loc[(preds["model"] == "bt") & (preds["mode"] == "live"), "match_id"]
    out = pd.concat([preds, forecasts[~forecasts["match_id"].isin(live)]], ignore_index=True)
    return out.sort_values("match_date_utc", kind="stable", ignore_index=True)


def live_scores(path: Path = paths.LIVE_SCORES_MD) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


# ---------------------------------------------------------------- views for a general audience

def region_share(odds: pd.DataFrame) -> pd.DataFrame:
    """Title chance summed per region (from title_odds), largest first, with display name."""
    if odds.empty:
        return pd.DataFrame(columns=["region", "region_name", "champion"])
    out = odds.groupby("region", as_index=False)["Champion"].sum().rename(columns={"Champion": "champion"})
    out["region_name"] = out["region"].map(lambda r: REGIONS.get(r, (r, ""))[0])
    return out.sort_values("champion", ascending=False, ignore_index=True)


def swiss_split(swiss: pd.DataFrame) -> pd.DataFrame:
    """Long form for an advance-vs-out bar: one row per team and final record. Advancing records
    are positive, eliminations negative, so the bar splits at zero. Order: 3-0 outward, 0-3 outward."""
    if swiss.empty:
        return pd.DataFrame(columns=["team", "name", "record", "p", "signed", "group", "order"])
    long = swiss.melt(id_vars=["team", "name", "qualify"], value_vars=RECORDS, var_name="record", value_name="p")
    long["group"] = long["record"].map(lambda r: "Advances" if r.startswith("3") else "Eliminated")
    long["signed"] = long["p"].where(long["group"] == "Advances", -long["p"])
    # stack order from the centre line outward: 3-2, 3-1, 3-0 to the right; 2-3, 1-3, 0-3 to the left
    long["order"] = long["record"].map({"3-2": 0, "3-1": 1, "3-0": 2, "2-3": 0, "1-3": 1, "0-3": 2})
    return long.sort_values(["qualify", "group", "order"], ascending=[False, True, True], ignore_index=True)


def match_cards(preds: pd.DataFrame, names: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per match from the production model's live prediction: favourite, chance, result.
    `called_it` is True/False once played, NA before."""
    cols = ["date", "round", "favourite", "underdog", "chance", "best_of", "result", "called_it"]
    if preds.empty:
        return pd.DataFrame(columns=cols)
    p = preds[(preds["model"] == "bt") & preds["mode"].isin(["live", "preview"])] if "model" in preds else preds
    names = names or {}
    rows = []
    for r in p.to_dict("records"):
        a_fav = r["p_series_a"] >= 0.5
        fav, dog = (r["team_a"], r["team_b"]) if a_fav else (r["team_b"], r["team_a"])
        played = isinstance(r.get("winner"), str) and r["winner"] != ""
        rnd = f"Swiss round {r['round']}" if r["stage"] == "swiss" else {"QF1": "Quarterfinal", "QF2": "Quarterfinal",
              "QF3": "Quarterfinal", "QF4": "Quarterfinal", "SF1": "Semifinal", "SF2": "Semifinal", "F": "Final"}.get(r["round"], r["round"])
        rows.append({
            "date": pd.Timestamp(r["match_date_utc"]).strftime("%b %d"), "round": rnd,
            "favourite": names.get(fav, fav), "underdog": names.get(dog, dog),
            "chance": r["p_series_a"] if a_fav else 1 - r["p_series_a"], "best_of": int(r["best_of"]),
            "result": (f"{names.get(r['winner'], r['winner'])} won {r['score']}" if played else "Not played yet"),
            "called_it": (r["winner"] == fav) if played else pd.NA,
        })
    return pd.DataFrame(rows, columns=cols)


def sim_runs(report: Path = paths.TOURNAMENT_ODDS_MD) -> int | None:
    """Number of simulated tournaments behind the current odds, from the report's simulation line."""
    import re

    if not report.exists():
        return None
    m = re.search(r"\*\*Simulation:\*\*\s*([\d,]+)\s+runs", report.read_text(encoding="utf-8"))
    return int(m.group(1).replace(",", "")) if m else None


def last_updated(odds_csv: Path = paths.TOURNAMENT_ODDS_CSV) -> pd.Timestamp | None:
    return pd.Timestamp(odds_csv.stat().st_mtime, unit="s", tz="UTC") if odds_csv.exists() else None
