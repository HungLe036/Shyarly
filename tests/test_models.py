"""Model behaviour and the hard no-leakage rule: changing or adding rows dated on or
after a prediction's cutoff day must not change that prediction."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from conftest import TRUE_REGION, make_games, make_queries, team_strength
from shyarly.models.baselines import CoinFlip
from shyarly.models.bt import BradleyTerry, BTParams
from shyarly.models.common import training_view
from shyarly.models.elo import Elo, EloParams

CUTOFF = pd.Timestamp("2025-10-01", tz="UTC")

BT = BTParams(half_life_days=3650, lambda_team=0.5, lambda_region_major=0.1,
              lambda_region_minor=0.1, intl_weight=1.0, lookback_days=3650, minor_regions=())


def all_models():
    return [
        CoinFlip(),
        BradleyTerry(BT),
        BradleyTerry(BT, use_teams=False),
        Elo(EloParams(k_team=24, k_region=8, season_carry=0.75)),
    ]


def test_training_view_excludes_cutoff_day(synth_games):
    day = pd.Timestamp("2025-03-01", tz="UTC")
    g = synth_games.copy()
    g.loc[0, "date_utc"] = day                              # exactly at the cutoff: excluded
    g.loc[1, "date_utc"] = day - pd.Timedelta(microseconds=1)  # just before: included
    tv = training_view(g, day, lookback_days=None)
    assert g.loc[0, "game_id"] not in set(tv["game_id"])
    assert g.loc[1, "game_id"] in set(tv["game_id"])
    assert (tv["date_utc"] < day).all()


@pytest.mark.parametrize("model", all_models(), ids=lambda m: m.name)
def test_future_rows_do_not_change_past_predictions(model, synth_games):
    queries = make_queries(synth_games, CUTOFF)
    before = model.predict(synth_games, queries)

    future = synth_games["date_utc"] >= CUTOFF
    changed = synth_games.copy()
    changed.loc[future, "blue_win"] = 1 - changed.loc[future, "blue_win"]          # flip every future result
    extra = make_games(seed=99, n_days=30)
    extra["date_utc"] = extra["date_utc"] + (CUTOFF - extra["date_utc"].min())    # brand-new future games
    extra["game_id"] = "x" + extra["game_id"]
    changed = pd.concat([changed, extra], ignore_index=True)

    after = model.predict(changed, queries)
    np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("model", all_models()[1:], ids=lambda m: m.name)
def test_past_rows_do_change_predictions(model, synth_games):
    """Guard against the leakage test passing vacuously."""
    queries = make_queries(synth_games, CUTOFF)
    past = synth_games["date_utc"] < CUTOFF
    changed = synth_games.copy()
    changed.loc[past, "blue_win"] = 1 - changed.loc[past, "blue_win"]
    assert not np.allclose(model.predict(synth_games, queries), model.predict(changed, queries))


def test_bt_recovers_regions_and_teams(synth_games):
    fit = BradleyTerry(BT).fit(synth_games, CUTOFF)
    r = fit.region
    assert r["A"] > r["B"] > r["C"]
    assert (r["A"] - r["C"]) == pytest.approx(TRUE_REGION["A"] - TRUE_REGION["C"], abs=0.5)
    teams = sorted({t for t in synth_games["blue_team"]})
    est = [fit.strength(f"{t.lower()}_2025", t[0]) for t in teams]
    assert spearmanr(est, [team_strength(t) for t in teams]).statistic > 0.9
    assert fit.beta == pytest.approx(0.1, abs=0.1)


def test_bt_is_antisymmetric_apart_from_side(synth_games):
    q = make_queries(synth_games, CUTOFF)
    swapped = q.rename(columns={
        "blue_team": "red_team", "red_team": "blue_team",
        "blue_team_season_id": "red_team_season_id", "red_team_season_id": "blue_team_season_id",
        "blue_region": "red_region", "red_region": "blue_region",
    })
    m = BradleyTerry(BT)
    fit = m.fit(synth_games, CUTOFF)
    z = np.log(m.predict(synth_games, q) / (1 - m.predict(synth_games, q)))
    zs = np.log(m.predict(synth_games, swapped) / (1 - m.predict(synth_games, swapped)))
    np.testing.assert_allclose(z - fit.beta, -(zs - fit.beta), atol=1e-9)


def test_unknown_team_gets_region_strength(synth_games):
    fit = BradleyTerry(BT).fit(synth_games, CUTOFF)
    assert fit.strength("nobody_2025", "A") == pytest.approx(fit.region["A"])
    assert fit.strength("nobody_2025", "Z") == 0.0


def test_elo_orders_regions(synth_games):
    q = pd.DataFrame({
        "cutoff": [CUTOFF], "date_utc": [CUTOFF], "blue_team": ["A0"], "red_team": ["C3"],
        "blue_team_season_id": ["a0_2025"], "red_team_season_id": ["c3_2025"],
        "blue_region": ["A"], "red_region": ["C"],
    })
    p = Elo(EloParams(k_team=24, k_region=8, season_carry=0.75)).predict(synth_games, q)
    assert p[0] > 0.8


def test_coinflip_is_half(synth_games):
    q = make_queries(synth_games, CUTOFF)
    assert (CoinFlip().predict(synth_games, q) == 0.5).all()
