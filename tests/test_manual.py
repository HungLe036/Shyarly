"""Hand-entered Worlds 2026 results: loading, validation and replay into simulator inputs."""

import random

import numpy as np
import pytest

from shyarly.config import load_leagues, load_worlds
from shyarly.data.manual import COLUMNS, ResultsError, knockout_fixed, load_results, swiss_fixed
from shyarly.sim.swiss import SwissRules, SwissSimulator

WORLDS = load_worlds()
LEAGUES = load_leagues()
CODES = [t.code for t in WORLDS.teams]
R1 = [tuple(p) for p in WORLDS.swiss["round1"]["pairings"]]


def write(tmp_path, rows):
    path = tmp_path / "results.csv"
    lines = [",".join(COLUMNS)]
    for r in rows:
        d = {"stage": "swiss", "date_utc": "2026-10-23T17:00:00Z", "winner": "", "score": "", "source": "test"} | r
        lines.append(",".join(str(d[c]) for c in COLUMNS))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_sim(**kw):
    return SwissSimulator(CODES, [LEAGUES.lookup(t.league, 2026)[0] for t in WORLDS.teams],
                          [t.tier for t in WORLDS.teams], np.full((16, 16), 0.5), SwissRules.from_config(WORLDS.swiss), **kw)


def played_swiss(seed=0, upto_round=5):
    """Rows for a plausible complete Swiss Stage, from one simulator run with the official Round 1."""
    log = make_sim(fixed_pairings={1: [(CODES.index(a), CODES.index(b)) for a, b in R1]}).run_once(random.Random(seed))
    rows = []
    for m in log.matches:
        if m.round > upto_round:
            continue
        need = (m.best_of + 1) // 2
        score = f"{need}-{need - 1}" if m.winner == m.a else f"{need - 1}-{need}"
        rows.append({"round": m.round, "team_a": CODES[m.a], "team_b": CODES[m.b], "winner": CODES[m.winner],
                     "score": score, "date_utc": f"2026-10-{22 + m.round}T17:00:00Z"})
    return rows, log


def test_missing_or_header_only_file_is_empty(tmp_path):
    assert load_results(tmp_path / "nope.csv", WORLDS) == []
    assert load_results(write(tmp_path, []), WORLDS) == []


def test_round1_rows_become_fixed_results(tmp_path):
    a, b = R1[0]
    rows = load_results(write(tmp_path, [{"round": 1, "team_a": a, "team_b": b, "winner": b, "score": "0-1"}]), WORLDS)
    fx = swiss_fixed(rows, WORLDS, CODES)
    assert fx.pairings[1] == [(CODES.index(x), CODES.index(y)) for x, y in R1]
    ia, ib = CODES.index(a), CODES.index(b)
    assert fx.results == {(1, min(ia, ib), max(ia, ib)): ib}
    assert fx.complete_rounds == 0
    assert fx.best_of[rows[0].match_id] == 1


def test_full_swiss_replays_and_fixes_outcome(tmp_path):
    rows, log = played_swiss(seed=3)
    fx = swiss_fixed(load_results(write(tmp_path, rows), WORLDS), WORLDS, CODES)
    assert fx.complete_rounds == 5
    assert fx.final_records == log.final
    res = make_sim(fixed_pairings=fx.pairings, fixed_results=fx.results).simulate(50, seed=1)
    np.testing.assert_array_equal(res.qualify, np.array([w == 3 for w, _ in log.final]) * 50)


@pytest.mark.parametrize("bad, msg", [
    ({"round": 1, "team_a": "GEN", "team_b": "XXX"}, "unknown team"),
    ({"round": 1, "team_a": "GEN", "team_b": "IG", "winner": "T1", "score": "1-0"}, "winner"),
    ({"round": 1, "team_a": "GEN", "team_b": "IG", "winner": "GEN", "score": "0-1"}, "score"),
    ({"round": 1, "team_a": "GEN", "team_b": "IG", "winner": "GEN", "score": ""}, "score"),
    ({"round": 6, "team_a": "GEN", "team_b": "IG"}, "round"),
    ({"round": 1, "team_a": "GEN", "team_b": "IG", "date_utc": "soon"}, "date"),
])
def test_bad_rows_are_rejected(tmp_path, bad, msg):
    with pytest.raises(ResultsError, match=msg):
        load_results(write(tmp_path, [bad]), WORLDS)


def test_round1_must_match_official_draw(tmp_path):
    (a, _), (_, d) = R1[0], R1[1]
    rows = load_results(write(tmp_path, [{"round": 1, "team_a": a, "team_b": d}]), WORLDS)
    with pytest.raises(ResultsError, match="official"):
        swiss_fixed(rows, WORLDS, CODES)


def test_best_of_must_follow_record(tmp_path):
    a, b = R1[0]
    rows = load_results(write(tmp_path, [{"round": 1, "team_a": a, "team_b": b, "winner": a, "score": "2-1"}]), WORLDS)
    with pytest.raises(ResultsError, match="Bo1"):
        swiss_fixed(rows, WORLDS, CODES)


def test_next_round_needs_previous_round_complete(tmp_path):
    rows, _ = played_swiss(seed=1, upto_round=2)
    r1 = [r for r in rows if r["round"] == 1]
    r1[0] = r1[0] | {"winner": "", "score": ""}
    rows = r1 + [r for r in rows if r["round"] == 2]
    with pytest.raises(ResultsError, match="Round 1"):
        swiss_fixed(load_results(write(tmp_path, rows), WORLDS), WORLDS, CODES)


def test_round_pairs_must_share_record(tmp_path):
    rows, _ = played_swiss(seed=2, upto_round=2)
    r1 = [r for r in rows if r["round"] == 1]
    winners = {r["winner"] for r in r1}
    r2 = [r | {"winner": "", "score": ""} for r in rows if r["round"] == 2]
    i = next(k for k, r in enumerate(r2) if r["team_a"] in winners)       # a 1-0 pair
    j = next(k for k, r in enumerate(r2) if r["team_a"] not in winners)   # a 0-1 pair
    r2[i]["team_b"], r2[j]["team_b"] = r2[j]["team_b"], r2[i]["team_b"]
    with pytest.raises(ResultsError, match="record"):
        swiss_fixed(load_results(write(tmp_path, r1 + r2), WORLDS), WORLDS, CODES)


def _ko_rows(log, swap_halves=False):
    by = {}
    for i, (w, l) in enumerate(log.final):
        by.setdefault(f"{w}-{l}", []).append(CODES[i])
    t30, t31, t32 = by["3-0"], by["3-1"], by["3-2"]
    qf = [(t30[0], t32[0]), (t31[0], t31[1]), (t30[1], t32[1]), (t31[2], t32[2])]
    if swap_halves:
        qf = [qf[0], qf[2], qf[1], qf[3]]
    return [{"stage": "knockout", "round": f"QF{k + 1}", "team_a": a, "team_b": b, "date_utc": "2026-11-03T22:00:00Z"}
            for k, (a, b) in enumerate(qf)]


def test_knockout_bracket_from_rows(tmp_path):
    rows, log = played_swiss(seed=4)
    ko = _ko_rows(log)
    ko[0] |= {"winner": ko[0]["team_a"], "score": "3-1"}
    loaded = load_results(write(tmp_path, rows + ko), WORLDS)
    kf = knockout_fixed(loaded, swiss_fixed(loaded, WORLDS, CODES), CODES)
    assert kf.bracket == [(CODES.index(r["team_a"]), CODES.index(r["team_b"])) for r in ko]
    assert kf.results == {"QF1": CODES.index(ko[0]["team_a"])}


def test_knockout_three_oh_teams_on_opposite_halves(tmp_path):
    rows, log = played_swiss(seed=4)
    loaded = load_results(write(tmp_path, rows + _ko_rows(log, swap_halves=True)), WORLDS)
    with pytest.raises(ResultsError, match="opposite"):
        knockout_fixed(loaded, swiss_fixed(loaded, WORLDS, CODES), CODES)


def test_knockout_needs_finished_swiss(tmp_path):
    rows, log = played_swiss(seed=4)
    loaded = load_results(write(tmp_path, [r for r in rows if r["round"] < 5] + _ko_rows(log)), WORLDS)
    with pytest.raises(ResultsError, match="Swiss"):
        knockout_fixed(loaded, swiss_fixed(loaded, WORLDS, CODES), CODES)
