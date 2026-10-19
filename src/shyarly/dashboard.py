"""Read-only loaders for the Streamlit page (`shyarly.app`). Pure pandas, no Streamlit import,
so they are testable without the `app` dependency group. Every loader returns an empty frame
(or None) when its file does not exist yet, e.g. before the first `shyarly update`."""

from pathlib import Path

import pandas as pd

from shyarly import paths

RECORDS = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]
STAGES = {"qualify": "Quarterfinals", "semis": "Semifinals", "final": "Final", "champion": "Champion"}


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
    keep = ["match_date_utc", "stage", "round", "team_a", "team_b", "best_of", "model", "mode",
            "p_series_a", "p_game_a", "winner", "score", "logged_at_utc"]
    return first[keep].sort_values(["match_date_utc", "model", "mode"], ignore_index=True)


def live_scores(path: Path = paths.LIVE_SCORES_MD) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None
