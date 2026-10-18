"""Sanity checks on config/worlds2026.yaml against the official ruleset structure."""

from collections import Counter
from itertools import permutations

import pytest

from shyarly import paths
from shyarly.config import load_sources, load_worlds

CFG = load_worlds()


def _record_counts_by_round(n_teams: int, n_rounds: int) -> list[Counter]:
    """Records (w, l) present at the start of each round. In a 16-team Swiss every
    same-record group has even size, so the counts are deterministic."""
    counts = Counter({(0, 0): n_teams})
    out = []
    for _ in range(n_rounds):
        active = Counter({r: c for r, c in counts.items() if 3 not in r})
        out.append(active)
        nxt = Counter({r: c for r, c in counts.items() if 3 in r})
        for (w, l), c in active.items():
            assert c % 2 == 0
            nxt[(w + 1, l)] += c // 2
            nxt[(w, l + 1)] += c // 2
        counts = nxt
    return out


def test_sixteen_slots_four_per_tier():
    assert len(CFG.teams) == CFG.swiss["n_teams"] == 16
    assert Counter(t.tier for t in CFG.teams) == {1: 4, 2: 4, 3: 4, 4: 4}
    assert len({t.code for t in CFG.teams}) == 16


def test_qualified_teams_have_known_league_and_seed():
    for t in CFG.teams:
        if t.status == "pending":
            assert t.candidates, "pending slot must list its candidates"
            continue
        assert t.league in CFG.regions, t
        assert t.seed is not None and 1 <= t.seed <= CFG.raw["berths"][t.league]


def test_tiers_match_ruleset_pools():
    # RULESET 4.3.2.1 defines pools by league seed.
    pools = {
        1: {("LCK", 1), ("LPL", 1), ("LCS", 1), ("LEC", 1)},
        2: {("LCP", 1), ("CBLOL", 1), ("LCK", 2), ("LPL", 2)},
        3: {("LCS", 2), ("LEC", 2), ("LCK", 3), ("LPL", 3)},
        4: {("LCP", 2), ("LPL", 4), ("LCK", 4)},  # + Play-In winner
    }
    for tier, expected in pools.items():
        got = {(t.league, t.seed) for t in CFG.teams if t.tier == tier and t.status != "pending"}
        assert got == expected


def test_pending_play_in_candidates_are_play_in_teams():
    pending = [t for t in CFG.teams if t.status == "pending"]
    assert len(pending) == 1
    play_in_codes = {t["code"] for t in CFG.raw["play_in"]["teams"]}
    assert set(pending[0].candidates) <= play_in_codes
    assert set(pending[0].candidates) == set(CFG.raw["play_in"]["final"]["teams"])


def _league_of(code: str) -> str:
    for t in CFG.raw["play_in"]["teams"]:
        if t["code"] == code:
            return t["league"]
    return CFG.team(code).league


@pytest.mark.parametrize("piw", CFG.team("PIW").candidates)
def test_round1_draw_without_same_region_exists(piw):
    leagues = {t.code: (t.league if t.status != "pending" else _league_of(piw)) for t in CFG.teams}
    by_tier = {k: [t.code for t in CFG.teams if t.tier == k] for k in (1, 2, 3, 4)}
    for a, b in CFG.swiss["round1"]["pool_matchups"]:
        ok = any(
            all(leagues[x] != leagues[y] for x, y in zip(by_tier[a], perm))
            for perm in permutations(by_tier[b])
        )
        assert ok, f"no valid pool {a} vs {b} draw with play-in winner {piw}"


def test_best_of_matches_ruleset_round_counts():
    # RULESET 4.1.2.1.1–5: R1 8xBo1, R2 8xBo1, R3 4xBo3 + 4xBo1, R4 6xBo3, R5 3xBo3
    expected = [{1: 8}, {1: 8}, {3: 4, 1: 4}, {3: 6}, {3: 3}]
    bo = CFG.swiss["best_of_by_record"]
    rounds = _record_counts_by_round(CFG.swiss["n_teams"], CFG.swiss["n_rounds"])
    for r, (active, exp) in enumerate(zip(rounds, expected), start=1):
        got = Counter()
        for (w, l), c in active.items():
            got[bo[f"{w}-{l}"]] += c // 2
        assert dict(got) == exp, f"round {r}"


def test_every_rule_block_cites_a_source():
    for block in ("event", "berths", "play_in", "swiss", "knockout"):
        refs = CFG.raw[block]["source"]
        assert refs, block
        for ref in refs:
            assert ref.split()[0] in CFG.raw["sources"], ref
    for url in CFG.raw["sources"].values():
        assert url.startswith("https://")


@pytest.mark.skipif(not paths.GAMES_PARQUET.exists(), reason="run `shyarly ingest` first")
def test_oe_names_play_in_their_league_in_2026():
    import pandas as pd

    g = pd.read_parquet(paths.GAMES_PARQUET, columns=["year", "league", "blue_team", "red_team"])
    g = g[g["year"] == 2026]
    played = set(zip(g["league"], g["blue_team"])) | set(zip(g["league"], g["red_team"]))
    teams = [t for t in CFG.teams if t.status != "pending"] + list(CFG.play_in_teams)
    for t in teams:
        assert t.oe_name, t.code
        assert (t.league, t.oe_name) in played, f"{t.code}: {t.oe_name!r} has no 2026 {t.league} games"


def test_candidate_takes_pending_slot_tier():
    for code in CFG.team("PIW").candidates:
        t = CFG.candidate(code)
        assert t.tier == CFG.team("PIW").tier and t.oe_name


def test_sources_yaml_has_all_backtest_years():
    s = load_sources()
    assert set(s.oracles_elixir.file_ids) == {2022, 2023, 2024, 2025, 2026}
