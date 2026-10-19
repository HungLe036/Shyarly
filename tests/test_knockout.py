"""Knockout bracket (RULESET 4.3.4) and the full-tournament simulation."""

import random
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from shyarly.config import load_leagues, load_worlds
from shyarly.data.manual import load_results
from shyarly.sim.knockout import FEEDS, QF_LABELS, draw_bracket, knockout_once
from shyarly.sim.run import simulate_worlds, tournament_markdown

WORLDS = load_worlds()
LEAGUES = load_leagues()


def records(seed):
    """A valid Swiss final standing: 2 at 3-0, 3 at 3-1, 3 at 3-2, 3 at 2-3, 3 at 1-3, 2 at 0-3."""
    recs = [(3, 0)] * 2 + [(3, 1)] * 3 + [(3, 2)] * 3 + [(2, 3)] * 3 + [(1, 3)] * 3 + [(0, 3)] * 2
    random.Random(seed).shuffle(recs)
    return recs


@pytest.mark.parametrize("seed", range(30))
def test_bracket_follows_ruleset(seed):
    final = records(seed)
    bracket = draw_bracket(final, random.Random(seed))
    teams = [t for m in bracket for t in m]
    assert sorted(teams) == sorted(i for i, (w, _) in enumerate(final) if w == 3)
    rec = lambda i: final[i]  # noqa: E731
    # 4.3.4.1: each 3-0 team meets a 3-2 team; 4.3.4.2: those two matches are on opposite halves.
    three_oh = [k for k, (a, b) in enumerate(bracket) if (3, 0) in (rec(a), rec(b))]
    assert len(three_oh) == 2
    for k in three_oh:
        assert sorted([rec(bracket[k][0]), rec(bracket[k][1])]) == [(3, 0), (3, 2)]
    halves = [{"QF1", "QF2"}, {"QF3", "QF4"}]
    labels = {QF_LABELS[k] for k in three_oh}
    assert all(len(labels & h) == 1 for h in halves)


def test_bracket_draw_is_random():
    final = records(0)
    seen = Counter(tuple(map(tuple, draw_bracket(final, random.Random(s)))) for s in range(400))
    assert len(seen) > 20


def test_knockout_has_one_champion_and_consistent_rounds():
    final = records(1)
    s = np.random.default_rng(0).normal(0, 1, 16)
    p5 = expit(s[:, None] - s[None, :]).tolist()
    for seed in range(50):
        bracket = draw_bracket(final, random.Random(seed))
        w = knockout_once(bracket, p5, random.Random(seed))
        for label, (x, y) in FEEDS.items():
            src = [w[x], w[y]]
            assert w[label] in src
        for k, q in enumerate(QF_LABELS):
            assert w[q] in bracket[k]


class StubPredictor:
    """Fixed strengths by team order; enough for simulate_worlds."""
    as_of = pd.Timestamp("2026-10-20", tz="UTC")

    def game_matrix(self, refs):
        s = np.linspace(1.5, -1.5, len(refs))
        return expit(s[:, None] - s[None, :])

    def game_matrix_draws(self, refs, n, rng):
        s = np.linspace(1.5, -1.5, len(refs)) + rng.normal(0, 0.3, (n, len(refs)))
        return expit(s[:, :, None] - s[:, None, :])


@pytest.mark.parametrize("posterior", [False, True])
def test_tournament_odds_add_up(posterior):
    odds = simulate_worlds(WORLDS, LEAGUES, StubPredictor(), runs=3000, seed=4, posterior=posterior, knockout=True)
    t = odds.table
    assert t["qualify"].sum() == pytest.approx(8)
    assert t["semis"].sum() == pytest.approx(4)
    assert t["final"].sum() == pytest.approx(2)
    assert t["champion"].sum() == pytest.approx(1)
    assert (t["champion"] <= t["final"]).all() and (t["final"] <= t["semis"]).all() and (t["semis"] <= t["qualify"]).all()
    assert "champion" in tournament_markdown(odds, StubPredictor.as_of, 1.0, StubPredictor.as_of)


def test_tournament_with_fixed_swiss_and_bracket(tmp_path):
    from test_manual import _ko_rows, played_swiss, write

    rows, log = played_swiss(seed=6)
    ko = _ko_rows(log)
    ko[0] |= {"winner": ko[0]["team_a"], "score": "3-2"}
    results = load_results(write(tmp_path, rows + ko), WORLDS)
    odds = simulate_worlds(WORLDS, LEAGUES, StubPredictor(), runs=500, seed=1, results=results, knockout=True)
    t = odds.table.set_index("team")
    codes = [x.code for x in WORLDS.teams]
    qualified = {codes[i] for i, (w, _) in enumerate(log.final) if w == 3}
    assert set(t.index[t["qualify"] == 1.0]) == qualified
    assert t.loc[ko[0]["team_a"], "semis"] == 1.0 and t.loc[ko[0]["team_b"], "semis"] == 0.0
    assert odds.bracket_fixed and odds.rounds_complete == 5 and odds.results_fixed == len(rows) + 1


def test_fixed_knockout_results_are_used():
    final = records(2)
    p5 = np.full((16, 16), 0.5).tolist()
    bracket = draw_bracket(final, random.Random(0))
    fixed = {"QF1": bracket[0][1], "QF2": bracket[1][0]}
    for seed in range(20):
        w = knockout_once(bracket, p5, random.Random(seed), fixed)
        assert w["QF1"] == bracket[0][1] and w["QF2"] == bracket[1][0]
