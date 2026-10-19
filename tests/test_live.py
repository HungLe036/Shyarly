"""Prediction log for Worlds 2026 matches (the fresh holdout) and its scoring."""

import math

import pandas as pd
import pytest

from conftest import make_games
from shyarly.config import load_worlds
from shyarly.data.manual import COLUMNS, load_results
from shyarly.eval.live import LOG_COLUMNS, append_log, due_predictions, score_log, stage_starts
from shyarly.models.bt import BTParams
from shyarly.models.common import day_start
from shyarly.models.predictor import Predictor, TeamRef
from shyarly.models.series import series_win_prob

WORLDS = load_worlds()
PARAMS = BTParams(half_life_days=90, half_life_intl_days=365, lambda_team=1.0, lambda_region_major=0.1,
                  lambda_region_minor=0.1, intl_weight=1.0, lookback_days=730)
GAMES = make_games(n_days=420)
SYNTH = {"GEN": "A0", "IG": "C3", "TLAW": "B1", "CFO": "C0"}
PROV = {"git_commit": "abc123", "config_hash": "deadbeef"}


def ref(code):
    t = SYNTH[code]
    return TeamRef(code, t, t[0])


def predictor_at(cutoff):
    return Predictor(GAMES, cutoff, PARAMS, temperature=1.0)


def rows(tmp_path, extra=()):
    lines = [",".join(COLUMNS),
             "swiss,1,GEN,IG,2026-10-23T17:00:00Z,,,t",
             "swiss,1,TLAW,CFO,2026-10-24T18:00:00Z,,,t", *extra]
    p = tmp_path / "r.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_results(p, WORLDS)


def best_of(rs):
    return {r.match_id: 1 for r in rs}


def test_nothing_is_logged_before_its_cutoff(tmp_path):
    rs = rows(tmp_path)
    out = due_predictions(rs, best_of(rs), predictor_at, ref, stage_starts(WORLDS),
                          now=pd.Timestamp("2026-10-22T12:00", tz="UTC"), provenance=PROV)
    assert out.empty and list(out.columns) == LOG_COLUMNS


def test_cutoffs_and_modes(tmp_path):
    rs = rows(tmp_path)
    now = pd.Timestamp("2026-10-23T09:00", tz="UTC")
    out = due_predictions(rs, best_of(rs), predictor_at, ref, stage_starts(WORLDS), now=now, provenance=PROV)
    # Oct 23 match: pre-event and live both due (cutoff Oct 23 00:00). Oct 24 match: only pre-event.
    got = set(zip(out["match_id"], out["mode"]))
    m1, m2 = rs[0].match_id, rs[1].match_id
    assert got == {(m1, "pre_event"), (m1, "live"), (m2, "pre_event")}
    assert (pd.to_datetime(out.loc[out["mode"] == "pre_event", "cutoff_utc"]) == stage_starts(WORLDS)["swiss"]).all()
    live = out[out["mode"] == "live"].iloc[0]
    assert pd.Timestamp(live["cutoff_utc"]) == day_start(rs[0].date_utc)
    assert live["p_series_a"] == pytest.approx(series_win_prob(live["p_game_a"], 1))
    assert (out["git_commit"] == "abc123").all() and (out["logged_at_utc"] == now.isoformat()).all()


def test_predictions_match_point_model(tmp_path):
    rs = rows(tmp_path)
    out = due_predictions(rs, best_of(rs), predictor_at, ref, stage_starts(WORLDS),
                          now=pd.Timestamp("2026-10-25", tz="UTC"), provenance=PROV)
    r = out[(out["match_id"] == rs[1].match_id) & (out["mode"] == "live")].iloc[0]
    expected = predictor_at(pd.Timestamp("2026-10-24", tz="UTC")).p_game(ref("TLAW"), ref("CFO"))
    assert r["p_game_a"] == pytest.approx(expected)


def test_matches_without_known_best_of_are_skipped(tmp_path):
    rs = rows(tmp_path)
    out = due_predictions(rs, {}, predictor_at, ref, stage_starts(WORLDS),
                          now=pd.Timestamp("2026-10-25", tz="UTC"), provenance=PROV)
    assert out.empty


def test_append_keeps_first_prediction(tmp_path):
    rs = rows(tmp_path)
    path = tmp_path / "log.csv"
    first = due_predictions(rs, best_of(rs), predictor_at, ref, stage_starts(WORLDS),
                            now=pd.Timestamp("2026-10-23T09:00", tz="UTC"), provenance=PROV)
    assert append_log(path, first) == 3
    later = due_predictions(rs, best_of(rs), predictor_at, ref, stage_starts(WORLDS),
                            now=pd.Timestamp("2026-10-25", tz="UTC"), provenance=PROV | {"git_commit": "new"})
    assert append_log(path, later) == 1          # only the Oct 24 live prediction is new
    log = pd.read_csv(path)
    assert len(log) == 4 and (log["git_commit"] == "abc123").sum() == 3


def test_models_are_logged_and_scored_separately(tmp_path):
    rs = rows(tmp_path)
    path = tmp_path / "log.csv"
    # an older log without the model column holds production rows
    pd.DataFrame(columns=[c for c in LOG_COLUMNS if c != "model"]).to_csv(path, index=False)
    now = pd.Timestamp("2026-10-23T09:00", tz="UTC")
    args = (rs, best_of(rs), predictor_at, ref, stage_starts(WORLDS))
    assert append_log(path, due_predictions(*args, now=now, provenance=PROV)) == 3
    assert append_log(path, due_predictions(*args, now=now, provenance=PROV, model="margin_bt")) == 3
    assert append_log(path, due_predictions(*args, now=now, provenance=PROV, model="margin_bt")) == 0
    log = pd.read_csv(path)
    assert list(log.columns) == LOG_COLUMNS
    assert sorted(log["model"].value_counts().items()) == [("bt", 3), ("margin_bt", 3)]
    played = [r.__class__(**{**r.__dict__, "winner": r.team_a, "score": (1, 0)}) for r in rs]
    s = score_log(log, played)
    assert set(zip(s["model"], s["mode"])) == {("bt", "pre_event"), ("bt", "live"),
                                               ("margin_bt", "pre_event"), ("margin_bt", "live")}


def test_score_log(tmp_path):
    rs = rows(tmp_path, extra=())
    played = [r.__class__(**{**r.__dict__, "winner": r.team_a if k == 0 else r.team_b,
                             "score": (1, 0) if k == 0 else (0, 1)}) for k, r in enumerate(rs)]
    log = pd.DataFrame([
        {"match_id": rs[0].match_id, "mode": "live", "p_game_a": 0.7, "p_series_a": 0.7, "best_of": 1},
        {"match_id": rs[1].match_id, "mode": "live", "p_game_a": 0.6, "p_series_a": 0.6, "best_of": 1},
        {"match_id": rs[1].match_id, "mode": "live", "p_game_a": 0.1, "p_series_a": 0.1, "best_of": 1},  # later dup: ignored
    ])
    s = score_log(log, played).set_index("mode").loc["live"]
    assert s["n_series"] == 2 and s["n_games"] == 2
    assert s["series_log_loss"] == pytest.approx(-(math.log(0.7) + math.log(0.4)) / 2)
    assert s["series_brier"] == pytest.approx((0.3**2 + 0.6**2) / 2)
    assert s["game_log_loss"] == pytest.approx(s["series_log_loss"])
