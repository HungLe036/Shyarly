"""Prediction log for Worlds 2026 matches: the fresh, untouched holdout.

Two predictions per match (spec M5, "pre-event vs live"):
  * pre_event: model trained on games before the stage starts (Swiss Oct 23, knockouts Nov 3);
  * live:      model trained on games before the match's UTC day.
A prediction is logged only once its cutoff has passed (so every game before the cutoff
could be in the data) and the log is append-only: the first row per (match, mode) is the
one that counts. Each row records the git commit and a hash of the model and tournament
configs, so a later model change cannot silently rewrite what was predicted.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from shyarly import paths
from shyarly.config import WorldsConfig
from shyarly.data.manual import KO_BEST_OF, MatchRow
from shyarly.models.common import day_start
from shyarly.models.predictor import Predictor, TeamRef
from shyarly.models.series import series_win_prob

LOG_COLUMNS = ["match_id", "stage", "round", "team_a", "team_b", "match_date_utc", "best_of", "mode",
               "cutoff_utc", "p_game_a", "p_series_a", "logged_at_utc", "git_commit", "config_hash"]
MODES = ("pre_event", "live")
SCORE_COLUMNS = ["mode", "n_series", "series_log_loss", "series_brier", "n_games", "game_log_loss"]


def stage_starts(worlds: WorldsConfig) -> dict[str, pd.Timestamp]:
    st = worlds.raw["event"]["stages"]
    as_utc = lambda d: pd.Timestamp(str(d), tz="UTC")  # noqa: E731
    return {"swiss": as_utc(st["swiss"]["start"]), "knockout": as_utc(st["knockout"]["quarterfinals"]["start"])}


def provenance(config_paths: tuple[Path, ...] = (paths.MODEL_CONFIG, paths.WORLDS_CONFIG)) -> dict[str, str]:
    h = hashlib.sha256()
    for p in config_paths:
        h.update(Path(p).read_bytes())
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "config"], cwd=paths.ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        commit += "-dirty" if dirty else ""
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {"git_commit": commit, "config_hash": h.hexdigest()[:12]}


def due_predictions(rows: list[MatchRow], best_of: dict[str, int], predictor_at: Callable[[pd.Timestamp], Predictor],
                    ref: Callable[[str], TeamRef], starts: dict[str, pd.Timestamp], now: pd.Timestamp,
                    provenance: dict[str, str]) -> pd.DataFrame:
    """Predictions whose cutoff has passed, for every match with a known pairing and best-of."""
    cache: dict[pd.Timestamp, Predictor] = {}
    out = []
    for r in rows:
        bo = KO_BEST_OF if r.stage == "knockout" else best_of.get(r.match_id)
        if bo is None:
            continue
        match_day = day_start(r.date_utc)
        for mode in MODES:
            cutoff = min(starts[r.stage], match_day) if mode == "pre_event" else match_day
            if now < cutoff:
                continue
            if cutoff not in cache:
                cache[cutoff] = predictor_at(cutoff)
            p_game = float(cache[cutoff].p_game(ref(r.team_a), ref(r.team_b)))
            out.append({
                "match_id": r.match_id, "stage": r.stage, "round": r.round, "team_a": r.team_a, "team_b": r.team_b,
                "match_date_utc": r.date_utc.isoformat(), "best_of": bo, "mode": mode,
                "cutoff_utc": cutoff.isoformat(), "p_game_a": p_game, "p_series_a": float(series_win_prob(p_game, bo)),
                "logged_at_utc": now.isoformat(), **provenance,
            })
    return pd.DataFrame(out, columns=LOG_COLUMNS)


def append_log(path: Path, new: pd.DataFrame) -> int:
    """Append rows for (match_id, mode) pairs not yet in the log. Returns how many were added."""
    path = Path(path)
    old = pd.read_csv(path, dtype={"round": str}) if path.exists() else pd.DataFrame(columns=LOG_COLUMNS)
    have = set(zip(old["match_id"], old["mode"]))
    add = new[[(m, md) not in have for m, md in zip(new["match_id"], new["mode"])]]
    if not path.exists() or len(add):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([old, add], ignore_index=True)[LOG_COLUMNS].to_csv(path, index=False)
    return len(add)


def score_log(log: pd.DataFrame, rows: list[MatchRow]) -> pd.DataFrame:
    """Log loss and Brier of the first logged prediction per (match, mode), on played matches.
    Game log loss uses the series score: each game is a Bernoulli trial with p_game."""
    played = {r.match_id: r for r in rows if r.winner is not None and r.score is not None}
    first = log.drop_duplicates(["match_id", "mode"], keep="first")
    out = []
    for mode, grp in first.groupby("mode", sort=False):
        ll, br, gll, n_games = [], [], 0.0, 0
        for rec in grp.to_dict("records"):
            r = played.get(rec["match_id"])
            if r is None:
                continue
            flip = "team_a" in rec and rec["team_a"] != r.team_a
            ps = float(np.clip(1 - rec["p_series_a"] if flip else rec["p_series_a"], 1e-6, 1 - 1e-6))
            pg = float(np.clip(1 - rec["p_game_a"] if flip else rec["p_game_a"], 1e-6, 1 - 1e-6))
            y = 1.0 if r.winner == r.team_a else 0.0
            ll.append(-(y * np.log(ps) + (1 - y) * np.log(1 - ps)))
            br.append((ps - y) ** 2)
            wa, wb = r.score
            gll += -(wa * np.log(pg) + wb * np.log(1 - pg))
            n_games += wa + wb
        if ll:
            out.append({"mode": mode, "n_series": len(ll), "series_log_loss": float(np.mean(ll)),
                        "series_brier": float(np.mean(br)), "n_games": n_games, "game_log_loss": gll / n_games})
    return pd.DataFrame(out, columns=SCORE_COLUMNS)


def live_markdown(scores: pd.DataFrame, log: pd.DataFrame, rows: list[MatchRow]) -> str:
    lines = ["# Worlds 2026 live scoring", "",
             "Predictions logged before each match (`reports/predictions_log.csv`, first row per match and mode "
             "counts), scored on played matches from `data/manual/worlds2026_results.csv`. "
             "pre_event = model trained on games before the stage started; live = before the match day. "
             "A coin flip scores 0.693 log loss and 0.25 Brier.", ""]
    if scores.empty:
        lines += ["No played matches with a logged prediction yet.", ""]
    else:
        lines += ["| mode | series | series log loss | series Brier | games | game log loss |", "|---|---|---|---|---|---|"]
        for s in scores.to_dict("records"):
            lines.append(f"| {s['mode']} | {s['n_series']} | {s['series_log_loss']:.4f} | {s['series_brier']:.4f} | "
                         f"{s['n_games']} | {s['game_log_loss']:.4f} |")
        lines.append("")
    played = {r.match_id: r for r in rows if r.winner}
    first = log.drop_duplicates(["match_id", "mode"], keep="first")
    if len(first):
        lines += ["## Logged predictions", "", "| match | Bo | mode | P(team_a wins) | result | logged at | commit |",
                  "|---|---|---|---|---|---|---|"]
        for rec in first.to_dict("records"):
            r = played.get(rec["match_id"])
            res = f"{r.winner} {r.score[0]}-{r.score[1]}" if r else "—"
            lines.append(f"| {rec['team_a']} vs {rec['team_b']} ({rec['stage']} {rec['round']}) | {rec['best_of']} | "
                         f"{rec['mode']} | {100 * rec['p_series_a']:.1f}% | {res} | {str(rec['logged_at_utc'])[:16]} | "
                         f"{rec['git_commit']} |")
        lines.append("")
    return "\n".join(lines)
