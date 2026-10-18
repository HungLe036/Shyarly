"""Sanity checks on config/worlds2026.yaml against the official ruleset structure."""

from collections import Counter
from itertools import permutations

import pytest

from shyarly import paths
from shyarly.config import Team, load_sources, load_worlds

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
    play_in_codes = {t.code for t in CFG.play_in_teams}
    for tier, expected in pools.items():
        got = {(t.league, t.seed) for t in CFG.teams
               if t.tier == tier and t.status != "pending" and t.code not in play_in_codes}
        assert got == expected
    assert sum(t.tier == 4 and (t.code in play_in_codes or t.status == "pending") for t in CFG.teams) == 1


def test_play_in_winner_fills_pool4_slot():
    final = CFG.raw["play_in"]["final"]
    play_in_codes = {t.code for t in CFG.play_in_teams}
    in_swiss = [t for t in CFG.teams if t.code in play_in_codes]
    if final["status"] == "pending":
        pending = [t for t in CFG.teams if t.status == "pending"]
        assert len(pending) == 1 and not in_swiss
        assert set(pending[0].candidates) == set(final["teams"]) <= play_in_codes
    else:
        assert final["winner"] in final["teams"]
        assert [t.code for t in in_swiss] == [final["winner"]] and in_swiss[0].tier == 4
        assert not any(t.status == "pending" for t in CFG.teams)
        assert set(CFG.raw["play_in"]["eliminated"]) == play_in_codes - {final["winner"]}


def test_round1_draw_without_same_region_exists():
    leagues = {t.code: t.league for t in CFG.teams if t.status != "pending"}
    by_tier = {k: [t.code for t in CFG.teams if t.tier == k] for k in (1, 2, 3, 4)}
    for a, b in CFG.swiss["round1"]["pool_matchups"]:
        ok = any(
            all(leagues.get(x) != leagues.get(y) for x, y in zip(by_tier[a], perm))
            for perm in permutations(by_tier[b])
        )
        assert ok, f"no valid pool {a} vs {b} draw"


def test_official_round1_pairings_follow_ruleset():
    pairs = CFG.swiss["round1"]["pairings"]
    if not pairs:
        pytest.skip("Round 1 draw not published yet")
    by_code = {t.code: t for t in CFG.teams}
    flat = [c for p in pairs for c in p]
    assert sorted(flat) == sorted(by_code), "each team exactly once"
    allowed = {frozenset(m) for m in CFG.swiss["round1"]["pool_matchups"]}
    for a, b in pairs:
        ta, tb = by_code[a], by_code[b]
        assert frozenset((ta.tier, tb.tier)) in allowed, (a, b)
        assert ta.league != tb.league, (a, b)


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
    # Exercise the pending-slot path on a copy where the Play-In slot is still open.
    from dataclasses import replace

    final = CFG.raw["play_in"]["final"]
    slot = Team(code="PIW", name="Play-In Winner", league=None, seed=None, tier=4,
                status="pending", candidates=tuple(final["teams"]))
    teams = tuple(slot if t.code in {p.code for p in CFG.play_in_teams} else t for t in CFG.teams)
    pending = replace(CFG, teams=teams) if CFG.raw["play_in"]["final"]["status"] != "pending" else CFG
    pw = next(t for t in pending.teams if t.status == "pending")
    for code in pw.candidates:
        t = pending.candidate(code)
        assert t.tier == pw.tier and t.oe_name


def test_sources_yaml_has_all_backtest_years():
    s = load_sources()
    assert set(s.oracles_elixir.file_ids) == {2022, 2023, 2024, 2025, 2026}
