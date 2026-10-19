"""Model vs betting market (closing Pinnacle odds) on matched series."""

import math

import numpy as np
import pandas as pd
import pytest

from conftest import make_games
from shyarly.eval.benchmark import add_model, join_market, name_map, score_benchmark
from shyarly.models.bt import BTParams


def market_rows():
    base = {"tournament": "T", "bookmaker": "pinnacle", "overround": 0.05}
    return pd.DataFrame([
        base | {"fixture_id": "f1", "start_time_utc": "2025-03-01T10:00:00+00:00", "team_1": "a0 esports",
                "team_2": "C3", "market": 181, "p1_fair": 0.7, "p2_fair": 0.3},
        base | {"fixture_id": "f1", "start_time_utc": "2025-03-01T10:00:00+00:00", "team_1": "a0 esports",
                "team_2": "C3", "market": 1847, "p1_fair": 0.6, "p2_fair": 0.4},
        base | {"fixture_id": "f2", "start_time_utc": "2025-03-02T10:00:00+00:00", "team_1": "B1",
                "team_2": "A1", "market": 181, "p1_fair": 0.45, "p2_fair": 0.55},
        base | {"fixture_id": "f3", "start_time_utc": "2025-03-05T10:00:00+00:00", "team_1": "Nobody",
                "team_2": "A1", "market": 181, "p1_fair": 0.5, "p2_fair": 0.5},
    ])


def series_rows():
    t = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    return pd.DataFrame([
        # stored with the teams in the other order: team_1 (A0) won 2-1
        {"series_id": "s1", "date_utc": t("2025-03-01T10:20"), "event": "E", "team_a": "C3", "team_b": "A0",
         "winner_team": "A0", "inferred_best_of": 3, "wins_a": 1, "wins_b": 2},
        {"series_id": "s2", "date_utc": t("2025-03-02T10:05"), "event": "E", "team_a": "B1", "team_b": "A1",
         "winner_team": "A1", "inferred_best_of": 1, "wins_a": 0, "wins_b": 1},
        # same pair a week later: must not match f2
        {"series_id": "s9", "date_utc": t("2025-03-09T10:00"), "event": "E", "team_a": "B1", "team_b": "A1",
         "winner_team": "B1", "inferred_best_of": 1, "wins_a": 1, "wins_b": 0},
    ])


def first_games():
    return pd.DataFrame([
        {"series_id": "s1", "game_in_series": 1, "blue_team": "C3", "red_team": "A0", "blue_win": 1},
        {"series_id": "s2", "game_in_series": 1, "blue_team": "A1", "red_team": "B1", "blue_win": 1},
    ])


def test_name_map_uses_normalisation_and_aliases():
    m, missing = name_map(["a0 esports", "TOP-esports", "Nobody"], {"A0", "Top Esports"}, {"a0 esports": "A0"})
    assert m == {"a0 esports": "A0", "TOP-esports": "Top Esports"} and missing == ["Nobody"]


def test_join_orients_outcomes_to_team_1():
    j, unmatched = join_market(market_rows(), series_rows(), first_games(), {"a0 esports": "A0"})
    j = j.set_index("fixture_id")
    assert set(j.index) == {"f1", "f2"} and unmatched == ["f3"]
    assert j.loc["f1", "team_1"] == "A0" and j.loc["f1", "series_id"] == "s1"
    assert j.loc["f1", "y_series"] == 1 and j.loc["f1", "y_map1"] == 0       # A0 lost game 1 (C3 on blue won)
    assert j.loc["f1", "p_mkt_series"] == 0.7 and j.loc["f1", "p_mkt_map1"] == 0.6
    assert j.loc["f2", "y_series"] == 0 and j.loc["f2", "y_map1"] == 0 and j.loc["f2", "best_of"] == 1
    assert np.isnan(j.loc["f2", "p_mkt_map1"])


def test_add_model_uses_games_before_the_match_day():
    games = make_games(n_days=200)
    params = BTParams(half_life_days=90, half_life_intl_days=365, lambda_team=1.0, lambda_region_major=0.1,
                      lambda_region_minor=0.1, intl_weight=1.0, lookback_days=730)
    j = pd.DataFrame([{"fixture_id": "x", "start": pd.Timestamp("2025-06-01T12:00", tz="UTC"),
                       "team_1": "A0", "team_2": "C3", "best_of": 3}])
    out = add_model(j, games, params, temperature=1.0)
    assert out.loc[0, "p_model_game"] > 0.7
    assert out.loc[0, "p_model_series"] == pytest.approx(
        out.loc[0, "p_model_game"] ** 2 * (3 - 2 * out.loc[0, "p_model_game"]))
    changed = games.copy()
    later = changed["date_utc"] >= pd.Timestamp("2025-06-01", tz="UTC")
    changed.loc[later, "blue_win"] = 1 - changed.loc[later, "blue_win"]
    assert add_model(j, changed, params, 1.0).loc[0, "p_model_game"] == out.loc[0, "p_model_game"]


def test_score_benchmark():
    j = pd.DataFrame({"fixture_id": ["a", "b"], "y_series": [1, 0], "p_model_series": [0.8, 0.4],
                      "p_mkt_series": [0.6, 0.3], "y_map1": [1, np.nan], "p_model_game": [0.7, 0.45],
                      "p_mkt_map1": [0.65, np.nan]})
    s = score_benchmark(j, resamples=200, seed=0).set_index("target")
    ser = s.loc["series"]
    assert ser["n"] == 2
    assert ser["model_log_loss"] == pytest.approx(-(math.log(0.8) + math.log(0.6)) / 2)
    assert ser["market_log_loss"] == pytest.approx(-(math.log(0.6) + math.log(0.7)) / 2)
    assert ser["delta"] == pytest.approx(ser["model_log_loss"] - ser["market_log_loss"])
    assert s.loc["first map", "n"] == 1
