"""Production predictor (calibrated BT at a point in time) and the end-to-end Swiss run."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_games
from shyarly import paths
from shyarly.config import load_leagues, load_model_config, load_worlds
from shyarly.models.bt import BTParams
from shyarly.models.predictor import Predictor, TeamRef
from shyarly.models.series import series_win_prob

AS_OF = pd.Timestamp("2025-10-01", tz="UTC")
PARAMS = BTParams(half_life_days=90, half_life_intl_days=365, lambda_team=1.0, lambda_region_major=0.1,
                  lambda_region_minor=0.1, intl_weight=1.0, lookback_days=730)
A0, C3 = TeamRef("A0", "A0", "A"), TeamRef("C3", "C3", "C")


@pytest.fixture(scope="module")
def games():
    return make_games()


def test_game_probability_is_symmetric_and_favours_stronger_team(games):
    pr = Predictor(games, AS_OF, PARAMS, temperature=1.0)
    assert pr.p_game(A0, C3) == pytest.approx(1 - pr.p_game(C3, A0))
    assert pr.p_game(A0, C3) > 0.8
    assert pr.p_series(A0, C3, 5) == pytest.approx(series_win_prob(pr.p_game(A0, C3), 5))


def test_temperature_pulls_toward_half(games):
    hot = Predictor(games, AS_OF, PARAMS, temperature=1.0).p_game(A0, C3)
    cool = Predictor(games, AS_OF, PARAMS, temperature=2.0).p_game(A0, C3)
    assert 0.5 < cool < hot


def test_game_matrix(games):
    pr = Predictor(games, AS_OF, PARAMS, temperature=1.0)
    m = pr.game_matrix([A0, C3, TeamRef("B1", "B1", "B")])
    np.testing.assert_allclose(m + m.T, 1.0)
    assert m[0, 1] == pytest.approx(pr.p_game(A0, C3))


def test_future_rows_do_not_change_predictions(games):
    base = Predictor(games, AS_OF, PARAMS, temperature=1.05).p_game(A0, C3)
    changed = games.copy()
    future = changed["date_utc"] >= AS_OF
    changed.loc[future, "blue_win"] = 1 - changed.loc[future, "blue_win"]
    assert Predictor(changed, AS_OF, PARAMS, temperature=1.05).p_game(A0, C3) == base


@pytest.mark.skipif(not paths.GAMES_PARQUET.exists(), reason="run `shyarly ingest` first")
def test_worlds_swiss_run_end_to_end():
    from shyarly.sim.run import simulate_worlds_swiss

    worlds, leagues, cfg = load_worlds(), load_leagues(), load_model_config()
    g = pd.read_parquet(paths.GAMES_PARQUET)
    pr = Predictor.from_config(g, pd.Timestamp("2026-10-18", tz="UTC"), cfg)
    odds = simulate_worlds_swiss(worlds, leagues, pr, runs=2000, seed=1)
    t = odds.table.set_index("team")
    assert t["qualify"].sum() == pytest.approx(8.0)
    assert t["in_swiss"].sum() == pytest.approx(16.0)
    pending = [x for x in worlds.teams if x.status == "pending"]
    if pending:
        assert sum(odds.play_in.values()) == pytest.approx(1.0)
        assert t.loc[list(pending[0].candidates), "in_swiss"].sum() == pytest.approx(1.0)
    assert (t["qualify"] <= t["in_swiss"] + 1e-12).all()
