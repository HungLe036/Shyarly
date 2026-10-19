"""Margin-augmented BT (shadow model margin_bt): gold difference at 15 minutes as extra evidence."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from conftest import make_games, team_strength
from shyarly.eval.shadow import fit_margin_scale
from shyarly.models.bt import BradleyTerry, BTParams

AS_OF = pd.Timestamp("2025-04-01", tz="UTC")
BASE = BTParams(half_life_days=90, half_life_intl_days=365, lambda_team=1.0, lambda_region_major=0.1,
                lambda_region_minor=0.1, intl_weight=1.0, lookback_days=730)
MARGIN = replace(BASE, margin_weight=1.0, margin_scale=2500.0, margin_sigma=1.0)
TEAMS = [f"{r}{i}" for r in "ABC" for i in range(4)]


@pytest.fixture(scope="module")
def gold_games():
    return make_games(seed=1, n_days=90, games_per_day=3, gold_per_logit=2500.0, gold_noise=2500.0)


def strengths(fit):
    return np.array([fit.strength(f"{t.lower()}_2025", t[0]) for t in TEAMS])


def truth():
    return np.array([team_strength(t) for t in TEAMS])


def test_zero_weight_is_plain_bt(gold_games):
    a = BradleyTerry(BASE).fit(gold_games, AS_OF)
    b = BradleyTerry(replace(MARGIN, margin_weight=0.0)).fit(gold_games, AS_OF)
    np.testing.assert_array_equal(a.theta, b.theta)


def test_missing_gold_only_uses_wins(gold_games):
    no_gold = gold_games.assign(gd15=np.nan)
    a = BradleyTerry(BASE).fit(no_gold, AS_OF)
    b = BradleyTerry(MARGIN).fit(no_gold, AS_OF)
    np.testing.assert_allclose(a.theta, b.theta, atol=1e-6)
    c = BradleyTerry(MARGIN).fit(gold_games.drop(columns="gd15"), AS_OF)
    np.testing.assert_allclose(a.theta, c.theta, atol=1e-6)


def test_gold_margin_improves_strength_estimates():
    # Few games: win/loss alone is noisy; the gold margin carries more information per game.
    err_plain, err_margin = [], []
    for seed in range(6):
        g = make_games(seed=seed, n_days=40, games_per_day=3, gold_per_logit=2500.0, gold_noise=2500.0)
        t = truth()
        for params, errs in ((BASE, err_plain), (MARGIN, err_margin)):
            s = strengths(BradleyTerry(params).fit(g, AS_OF))
            errs.append(np.mean(((s - s.mean()) - (t - t.mean())) ** 2))
    assert np.mean(err_margin) < 0.7 * np.mean(err_plain)


def test_gradient_and_covariance_include_margin(gold_games):
    g = gold_games[gold_games["date_utc"] < "2025-02-01"]
    model = BradleyTerry(MARGIN)
    fit = model.fit(g, AS_OF, with_cov=True)
    loss, grad = model.objective(g, AS_OF)
    theta = fit.theta + 0.01
    eps = 1e-6
    num = np.array([(loss(theta + e) - loss(theta - e)) / (2 * eps) for e in np.eye(len(theta)) * eps])
    np.testing.assert_allclose(grad(theta), num, rtol=1e-4, atol=1e-5)
    h = np.array([(grad(fit.theta + e) - grad(fit.theta - e)) / (2e-5) for e in np.eye(len(theta)) * 1e-5]).T
    np.testing.assert_allclose(fit.cov, np.linalg.inv((h + h.T) / 2), rtol=1e-3, atol=1e-6)


def test_future_gold_does_not_change_fit(gold_games):
    a = BradleyTerry(MARGIN).fit(gold_games, AS_OF)
    changed = gold_games.copy()
    future = changed["date_utc"] >= AS_OF
    changed.loc[future, "gd15"] = -changed.loc[future, "gd15"] * 3
    b = BradleyTerry(MARGIN).fit(changed, AS_OF)
    np.testing.assert_array_equal(a.theta, b.theta)


def test_fit_margin_scale_recovers_gold_per_logit():
    g = make_games(seed=2, n_days=300, gold_per_logit=3000.0, gold_noise=1500.0)
    c, sigma, n = fit_margin_scale(g, BASE, pd.Timestamp("2025-10-01", tz="UTC"))
    # In-sample ridge logits are shrunk toward 0, so c comes out at or a bit above the truth.
    assert 2700 < c < 4000
    assert 0.2 < sigma < 1.5 and n > 1000
