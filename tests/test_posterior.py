"""Laplace posterior of the BT fit and the posterior-sampled Swiss simulation."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from conftest import make_games
from shyarly import paths
from shyarly.config import load_leagues, load_model_config, load_worlds
from shyarly.models.bt import BradleyTerry, BTParams
from shyarly.models.predictor import Predictor, TeamRef
from shyarly.sim.swiss import SwissRules, SwissSimulator

AS_OF = pd.Timestamp("2025-10-01", tz="UTC")
PARAMS = BTParams(half_life_days=90, half_life_intl_days=365, lambda_team=1.0, lambda_region_major=0.1,
                  lambda_region_minor=0.1, intl_weight=1.0, lookback_days=730)
TEAMS = [TeamRef(t, t, t[0]) for t in ("A0", "A3", "B1", "C3")]


@pytest.fixture(scope="module")
def games():
    return make_games(n_days=300)


def test_covariance_is_symmetric_positive_definite(games):
    fit = BradleyTerry(PARAMS).fit(games, AS_OF, with_cov=True)
    assert fit.cov.shape == (len(fit.index),) * 2
    np.testing.assert_allclose(fit.cov, fit.cov.T, atol=1e-12)
    assert np.linalg.eigvalsh(fit.cov).min() > 0


def test_point_fit_unchanged_by_covariance(games):
    a = BradleyTerry(PARAMS).fit(games, AS_OF)
    b = BradleyTerry(PARAMS).fit(games, AS_OF, with_cov=True)
    assert a.cov is None
    assert a.team == b.team and a.region == b.region and a.beta == b.beta


def test_covariance_matches_inverse_numeric_hessian():
    g = make_games(n_days=40, games_per_day=4)
    model = BradleyTerry(PARAMS)
    fit = model.fit(g, AS_OF, with_cov=True)
    loss, grad = model.objective(g, AS_OF)
    theta = fit.theta
    eps = 1e-5
    h = np.empty((len(theta), len(theta)))
    for i in range(len(theta)):
        d = np.zeros_like(theta)
        d[i] = eps
        h[:, i] = (grad(theta + d) - grad(theta - d)) / (2 * eps)
    np.testing.assert_allclose(fit.cov, np.linalg.inv((h + h.T) / 2), rtol=1e-3, atol=1e-6)


def test_more_evidence_shrinks_uncertainty(games):
    doubled = pd.concat([games, games.assign(game_id=games["game_id"] + "_dup")], ignore_index=True)
    # Only strength differences are identified by games (a region shift can be offset by its
    # teams), so compare the variance of a contrast: strength(A0) - strength(C3).
    def gap_var(fit):
        a = np.zeros(len(fit.index))
        for key, sign in ((("region", "A"), 1), (("team", "a0_2025"), 1), (("region", "C"), -1), (("team", "c3_2025"), -1)):
            a[fit.index[key]] += sign
        return a @ fit.cov @ a

    one = BradleyTerry(PARAMS).fit(games, AS_OF, with_cov=True)
    two = BradleyTerry(PARAMS).fit(doubled, AS_OF, with_cov=True)
    assert gap_var(two) < 0.75 * gap_var(one)


def test_draws_center_on_point_strengths(games):
    pr = Predictor(games, AS_OF, PARAMS, temperature=1.0, posterior=True)
    mean, cov = pr.strength_posterior(TEAMS)
    assert mean[0] == pytest.approx(pr.fit.beta)
    np.testing.assert_allclose(mean[1:], [pr.strength(t) for t in TEAMS])
    assert np.all(np.diag(cov) > 0)


def test_game_matrix_draws(games):
    pr = Predictor(games, AS_OF, PARAMS, temperature=1.05, posterior=True)
    draws = pr.game_matrix_draws(TEAMS, 4000, np.random.default_rng(0))
    assert draws.shape == (4000, 4, 4)
    np.testing.assert_allclose(draws + draws.transpose(0, 2, 1), 1.0)
    point = pr.game_matrix(TEAMS)
    mean = draws.mean(axis=0)
    # Averaging over strength uncertainty pulls game probabilities toward 0.5, never away.
    assert np.all(np.abs(mean - 0.5) <= np.abs(point - 0.5) + 2e-3)
    assert np.abs(mean - point).max() < 0.05


def test_game_matrix_draws_need_posterior(games):
    pr = Predictor(games, AS_OF, PARAMS, temperature=1.0)
    with pytest.raises(ValueError):
        pr.game_matrix_draws(TEAMS, 10, np.random.default_rng(0))


def test_future_rows_do_not_change_posterior(games):
    base = Predictor(games, AS_OF, PARAMS, 1.05, posterior=True).game_matrix_draws(TEAMS, 50, np.random.default_rng(3))
    changed = games.copy()
    future = changed["date_utc"] >= AS_OF
    changed.loc[future, "blue_win"] = 1 - changed.loc[future, "blue_win"]
    again = Predictor(changed, AS_OF, PARAMS, 1.05, posterior=True).game_matrix_draws(TEAMS, 50, np.random.default_rng(3))
    np.testing.assert_array_equal(base, again)


# ------------------------------------------------------------------ simulator

def _sim():
    worlds, leagues = load_worlds(), load_leagues()
    teams = list(worlds.teams)
    s = np.random.default_rng(1).normal(0, 1, 16)
    p = expit(s[:, None] - s[None, :])
    sim = SwissSimulator([t.code for t in teams], [leagues.lookup(t.league, 2026)[0] for t in teams],
                         [t.tier for t in teams], p, SwissRules.from_config(worlds.swiss))
    return sim, p


def test_constant_sampler_reproduces_point_simulation():
    sim, p = _sim()
    point = sim.simulate(500, seed=7)
    sampled = sim.simulate(500, seed=7, sampler=lambda n: np.broadcast_to(p, (n, 16, 16)))
    np.testing.assert_array_equal(point.qualify, sampled.qualify)
    np.testing.assert_array_equal(point.records, sampled.records)


def test_sampled_simulation_keeps_invariants():
    sim, p = _sim()
    rng = np.random.default_rng(2)

    def sampler(n):
        s = rng.normal(0, 1, (n, 16))
        return expit(s[:, :, None] - s[:, None, :])

    res = sim.simulate(3000, seed=1, sampler=sampler, chunk=700)
    assert res.qualify.sum() == 8 * 3000
    assert res.records.sum() == 16 * 3000
    assert np.all(res.records[:, :3].sum(axis=1) == res.qualify)


@pytest.mark.skipif(not paths.GAMES_PARQUET.exists(), reason="run `shyarly ingest` first")
def test_worlds_posterior_run_end_to_end():
    from shyarly.sim.run import simulate_worlds_swiss

    worlds, leagues, cfg = load_worlds(), load_leagues(), load_model_config()
    g = pd.read_parquet(paths.GAMES_PARQUET)
    pr = Predictor.from_config(g, pd.Timestamp("2026-10-18", tz="UTC"), cfg, posterior=True)
    odds = simulate_worlds_swiss(worlds, leagues, pr, runs=2000, seed=1, posterior=True)
    t = odds.table.set_index("team")
    assert odds.posterior
    assert t["qualify"].sum() == pytest.approx(8.0)
    assert (t["qualify"] <= t["in_swiss"] + 1e-12).all()
